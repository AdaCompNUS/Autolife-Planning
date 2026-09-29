"""TRAC-IK based inverse kinematics solver.

Wraps the vendored TRAC-IK C++ library (dual KDL + NLopt threading)
via pybind11 bindings. Pinocchio is used to compute the base-link-to-world
transform so that FK/IK targets use the same world frame as the Pink solver.
"""

from __future__ import annotations

import importlib

import numpy as np
from scipy.spatial.transform import Rotation

from autolife_planning.kinematics.ik_solver_base import IKSolverBase
from autolife_planning.types import (
    ChainConfig,
    IKConfig,
    IKResult,
    IKStatus,
    SE3Pose,
    SolveType,
)

pin = importlib.import_module("pinocchio")
if not hasattr(pin, "buildModelFromUrdf"):
    mod_path = getattr(pin, "__file__", "<unknown>")
    raise ModuleNotFoundError(
        "Imported 'pinocchio' does not expose Pinocchio robotics APIs "
        "(missing buildModelFromUrdf). This usually means a different PyPI "
        "package named 'pinocchio' was imported instead of the robotics one "
        "('pin').\n"
        f"Loaded module: {mod_path}\n"
        "Fix: uninstall the wrong package (`pip uninstall pinocchio`) and "
        "install the robotics package (`pip install pin`), or run with "
        "a clean PYTHONPATH/PYTHONNOUSERSITE."
    )

# Map our SolveType enum to pytracik C++ enum values
_SOLVE_TYPE_MAP = {
    SolveType.SPEED: "Speed",
    SolveType.DISTANCE: "Distance",
    SolveType.MANIP1: "Manip1",
    SolveType.MANIP2: "Manip2",
}


class TracIKSolver(IKSolverBase):
    """IK solver wrapping the TRAC-IK C++ library."""

    def __init__(
        self,
        chain_config: ChainConfig,
        config: IKConfig | None = None,
    ) -> None:
        if config is None:
            config = IKConfig()

        import pytracik

        # Read URDF string for TRAC-IK constructor
        with open(chain_config.urdf_path) as f:
            urdf_string = f.read()

        # Map SolveType enum
        cpp_solve_type = getattr(pytracik.SolveType, _SOLVE_TYPE_MAP[config.solve_type])

        self._trac_ik = pytracik.TRAC_IK(
            chain_config.base_link,
            chain_config.ee_link,
            urdf_string,
            config.timeout,
            config.epsilon,
            cpp_solve_type,
        )
        self._pytracik = pytracik
        self._chain_config = chain_config
        self._config = config

        # Cache joint limits
        self._lower_bounds = np.array(pytracik.get_joint_lower_bounds(self._trac_ik))
        self._upper_bounds = np.array(pytracik.get_joint_upper_bounds(self._trac_ik))
        self._n_joints = pytracik.get_num_joints(self._trac_ik)

        # Compute the base-link-to-world transform so FK/IK use world frame
        # (TRAC-IK's KDL chain returns poses relative to base_link).
        model = pin.buildModelFromUrdf(chain_config.urdf_path)
        data = model.createData()
        pin.forwardKinematics(model, data, pin.neutral(model))
        pin.updateFramePlacements(model, data)
        base_fid = model.getFrameId(chain_config.base_link)
        oMb = data.oMf[base_fid]
        self._base_R: np.ndarray = np.array(oMb.rotation, dtype=np.float64)
        self._base_t: np.ndarray = np.array(oMb.translation, dtype=np.float64)

    @property
    def base_frame(self) -> str:
        return self._chain_config.base_link

    @property
    def ee_frame(self) -> str:
        return self._chain_config.ee_link

    @property
    def num_joints(self) -> int:
        return self._n_joints

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower_bounds, upper_bounds) arrays."""
        return self._lower_bounds.copy(), self._upper_bounds.copy()

    def set_joint_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Override joint limits for the solver.

        Input:
            lower: Lower bounds array of length num_joints
            upper: Upper bounds array of length num_joints
        """
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if len(lower) != self._n_joints or len(upper) != self._n_joints:
            raise ValueError(
                f"Expected {self._n_joints} limits, "
                f"got lower={len(lower)}, upper={len(upper)}"
            )
        self._pytracik.set_joint_limits(self._trac_ik, lower, upper)
        self._lower_bounds = lower.copy()
        self._upper_bounds = upper.copy()

    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        """Compute forward kinematics in world frame.

        Input:
            joint_positions: Joint angles array of length num_joints
        Output:
            SE3Pose of the end effector in world frame
        """
        q = np.asarray(joint_positions, dtype=np.float64)
        T = self._pytracik.fk(self._trac_ik, q)
        if T.size == 0:
            raise RuntimeError("FK failed — invalid joint configuration")
        local = SE3Pose.from_matrix(T)
        return SE3Pose(
            position=self._base_R @ local.position + self._base_t,
            rotation=self._base_R @ local.rotation,
        )

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: IKConfig | None = None,
    ) -> IKResult:
        """Solve IK with random restart loop.

        Input:
            target_pose: Desired end-effector pose
            seed: Initial joint configuration (random if None)
            config: Override default IK config for this solve
        Output:
            IKResult with solution status and joint positions
        """
        cfg = config if config is not None else self._config

        # Convert world-frame target to base-link frame for pytracik
        R_inv = self._base_R.T
        local_pos = R_inv @ (target_pose.position - self._base_t)
        local_rot = R_inv @ target_pose.rotation
        x, y, z = local_pos
        quat_xyzw = Rotation.from_matrix(local_rot).as_quat()  # [x,y,z,w]
        qx, qy, qz, qw = quat_xyzw

        best_result: IKResult | None = None

        for attempt in range(cfg.max_attempts):
            if seed is not None and attempt == 0:
                q_seed = np.asarray(seed, dtype=np.float64)
            else:
                # Random seed within joint limits
                q_seed = np.random.uniform(self._lower_bounds, self._upper_bounds)

            result_arr = self._pytracik.ik(
                self._trac_ik, q_seed, x, y, z, qx, qy, qz, qw
            )

            ret_code = int(result_arr[0])
            if ret_code >= 0:
                q_solution = result_arr[1:]

                # Validate solution via FK
                achieved_pose = self.fk(q_solution)
                pos_err = float(
                    np.linalg.norm(achieved_pose.position - target_pose.position)
                )
                # Orientation error: angle of rotation difference
                R_err = achieved_pose.rotation.T @ target_pose.rotation
                ori_err = float(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec()))

                result = IKResult(
                    status=IKStatus.SUCCESS,
                    joint_positions=q_solution,
                    final_error=pos_err + ori_err,
                    iterations=attempt + 1,
                    position_error=pos_err,
                    orientation_error=ori_err,
                )

                # Check if within post-solve tolerances
                if (
                    pos_err < cfg.position_tolerance
                    and ori_err < cfg.orientation_tolerance
                ):
                    return result

                # Keep best result so far
                if best_result is None or result.final_error < best_result.final_error:
                    best_result = result

        # Return best result found, or failure
        if best_result is not None:
            return best_result

        return IKResult(
            status=IKStatus.FAILED,
            joint_positions=None,
            final_error=float("inf"),
            iterations=cfg.max_attempts,
            position_error=float("inf"),
            orientation_error=float("inf"),
        )
