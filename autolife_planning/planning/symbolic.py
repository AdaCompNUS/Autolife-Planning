"""Symbolic forward-kinematics helper shared by :class:`Constraint` and
:class:`Cost`.

Both user-defined constraints and costs build their CasADi expressions
on top of robot FK.  This module provides :class:`SymbolicContext`
which wraps either ``pinocchio.casadi`` (preferred — full-feature) or
``urdf2casadi`` (fallback — pure-Python, no planar base, no Jacobian
shortcuts) as the backend that turns the subgroup's active joint
symbol into symbolic link poses.

The context also owns small filesystem helpers shared across the
compile pipeline:

- :func:`_jit_build_dir` — scratch dir for CasADi's temporary JIT C files.
- :func:`_cwd` — ``chdir`` context manager used by CasADi's ``generate``,
  which always emits files into the process's current working directory.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import casadi as ca
import numpy as np

try:
    import pinocchio as pin
except Exception:  # pragma: no cover - optional runtime dependency
    pin = None
try:
    import pinocchio.casadi as cpin
except Exception:  # pragma: no cover - optional runtime dependency
    cpin = None
try:
    from urdf2casadi.urdfparser import URDFparser
except Exception:  # pragma: no cover - optional runtime dependency
    URDFparser = None

from autolife_planning.autolife import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    autolife_robot_config,
)


def _jit_build_dir() -> Path:
    """Directory for CasADi/urdf2casadi temporary JIT artifacts."""
    return (Path(__file__).resolve().parents[2] / "build" / "casadi_jit").resolve()


@contextmanager
def _cwd(path: Path):
    """Temporarily chdir — CasADi's generate() always writes to cwd."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class SymbolicContext:
    """CasADi-friendly view of the planner's active subgroup.

    Holds the CasADi symbolic joint vector ``q`` matching the subgroup's
    active dimension, plus a ``pinocchio.casadi`` model for symbolic FK.

    The context hides the planar-root encoding (the 24-DOF body vector
    uses ``[x, y, theta, j0..j20]``; pinocchio uses
    ``[x, y, cos(theta), sin(theta), j0..j20]``) and the mapping from
    the active subspace back to the full 24-DOF body via ``base_config``.
    """

    def __init__(
        self,
        subgroup: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        if base_config is None:
            base_config = HOME_JOINTS
        self.base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self.base_config.shape != HOME_JOINTS.shape:
            raise ValueError(
                f"base_config must have shape {HOME_JOINTS.shape}, "
                f"got {self.base_config.shape}"
            )

        self.subgroup_name = subgroup
        full_names = list(autolife_robot_config.joint_names)
        if subgroup == "autolife":
            self.active_indices = list(range(24))
            self.active_names = full_names
        else:
            sg = PLANNING_SUBGROUPS.get(subgroup)
            if sg is None:
                raise ValueError(f"Unknown subgroup: {subgroup!r}")
            self.active_names = list(sg["joints"])
            self.active_indices = [full_names.index(j) for j in self.active_names]

        self.q = ca.SX.sym("q", len(self.active_indices))
        self._full_names = list(autolife_robot_config.joint_names)
        self._root_link = "Link_Zero_Point"
        self._uses_planar_base = any(
            n in {"Joint_Virtual_X", "Joint_Virtual_Y", "Joint_Virtual_Theta"}
            for n in self.active_names
        )

        self._fk_cache: dict[str, dict[str, object]] = {}

        if pin is not None and cpin is not None:
            self._backend = "pinocchio"
            # Numeric + symbolic pinocchio models with planar root.
            self.pin_model = pin.buildModelFromUrdf(
                autolife_robot_config.urdf_path, pin.JointModelPlanar()
            )
            self.pin_data = self.pin_model.createData()
            self.cmodel = cpin.Model(self.pin_model)
            self.cdata = self.cmodel.createData()

            # Build the symbolic mapping q_active -> pinocchio q (nq=25).
            self.q_pin = self._build_pinocchio_q(self.q)

            # Pre-run symbolic FK so users can access frame transforms.
            cpin.forwardKinematics(self.cmodel, self.cdata, self.q_pin)
            cpin.updateFramePlacements(self.cmodel, self.cdata)
        elif URDFparser is not None:
            if self._uses_planar_base:
                raise RuntimeError(
                    "SymbolicContext fallback backend (urdf2casadi) does not support "
                    "planar base symbolic joints (Joint_Virtual_X/Y/Theta). "
                    "Install pinocchio + pinocchio.casadi for base-enabled groups."
                )
            self._backend = "urdf2casadi"
            self._urdf_parser = URDFparser()
            self._urdf_parser.from_file(autolife_robot_config.urdf_path)
            # Use the same world-like root as PyBullet visualization when available.
            try:
                self._urdf_parser.get_joint_info("Link_Zero_Point", "Link_Zero_Point")
                self._root_link = "Link_Zero_Point"
            except Exception:
                self._root_link = "Link_Ground_Vehicle"
        else:
            raise ModuleNotFoundError(
                "SymbolicContext requires either pinocchio.casadi or urdf2casadi. "
                "Install one of these backends."
            )

    def _build_pinocchio_q(self, q_active: ca.SX) -> ca.SX:
        """Map active-dim symbol to pinocchio q (nq=25)."""
        full: list[ca.SX | ca.DM] = [ca.DM(float(v)) for v in self.base_config]
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active[i]
        # full is 24 entries: [x, y, theta, j0..j20]
        pin_q = ca.vertcat(
            full[0],
            full[1],
            ca.cos(full[2]),
            ca.sin(full[2]),
            *full[3:],
        )
        return pin_q

    def _build_full_q(self, q_active: ca.SX) -> list[ca.SX | ca.DM]:
        """Map active subgroup symbols onto full 24-DOF joint vector."""
        full: list[ca.SX | ca.DM] = [ca.DM(float(v)) for v in self.base_config]
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active[i]
        return full

    def _urdf2casadi_pose(self, link_name: str, q_active: ca.SX):
        """Return symbolic link pose using urdf2casadi backend."""
        if link_name == self._root_link:
            T_expr = ca.SX.eye(4)
        else:
            cache = self._fk_cache.get(link_name)
            if cache is None:
                jit_dir = _jit_build_dir()
                jit_dir.mkdir(parents=True, exist_ok=True)
                with _cwd(jit_dir):
                    fk = self._urdf_parser.get_forward_kinematics(
                        self._root_link, link_name
                    )
                cache = {
                    "T_fk": fk["T_fk"],
                    "q_fk": fk["q"],
                    "joint_names": list(fk["joint_names"]),
                }
                # Warm up once under build/casadi_jit so CasADi's JIT-generated
                # temporary C files are created there instead of the caller's cwd.
                if isinstance(cache["T_fk"], ca.Function):
                    n_q = int(cache["q_fk"].numel())
                    with _cwd(jit_dir):
                        cache["T_fk"](np.zeros(n_q))
                self._fk_cache[link_name] = cache

            full = self._build_full_q(q_active)
            q_sub = []
            for joint_name in cache["joint_names"]:
                if joint_name not in self._full_names:
                    raise ValueError(
                        f"Joint {joint_name!r} from URDF chain is not in full joint list."
                    )
                q_sub.append(full[self._full_names.index(joint_name)])
            q_sub_expr = ca.vertcat(*q_sub) if q_sub else ca.SX([])
            T_fk = cache["T_fk"]
            if isinstance(T_fk, ca.Function):
                T_expr = T_fk(q_sub_expr)
            else:
                T_expr = ca.substitute(T_fk, cache["q_fk"], q_sub_expr)

        class _Pose:
            def __init__(self, T):
                self.translation = T[:3, 3]
                self.rotation = T[:3, :3]

        return _Pose(T_expr)

    def link_pose(self, link_name: str, q_active: ca.SX | None = None):
        """Return the symbolic pinocchio SE3 of a URDF link.

        Pass ``q_active=self.q`` (or omit) to get an expression that
        depends on the active joints symbolically.
        """
        if self._backend == "pinocchio":
            if q_active is None or q_active is self.q:
                frame_id = self.cmodel.getFrameId(link_name)
                return self.cdata.oMf[frame_id]
            # Rebuild with a different symbol.
            cdata = self.cmodel.createData()
            q_pin = self._build_pinocchio_q(q_active)
            cpin.forwardKinematics(self.cmodel, cdata, q_pin)
            cpin.updateFramePlacement(
                self.cmodel, cdata, self.cmodel.getFrameId(link_name)
            )
            return cdata.oMf[self.cmodel.getFrameId(link_name)]

        # urdf2casadi fallback
        q_eval = self.q if q_active is None else q_active
        return self._urdf2casadi_pose(link_name, q_eval)

    def link_translation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3-vector: link position in world frame."""
        return self.link_pose(link_name, q_active).translation

    def link_rotation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3x3 rotation matrix of the link."""
        return self.link_pose(link_name, q_active).rotation

    def evaluate_link_pose(
        self, link_name: str, q_active_numeric: np.ndarray
    ) -> np.ndarray:
        """Compute a NUMERIC 4x4 link pose (handy for building targets).

        Uses the numeric pinocchio model, not the symbolic one, so it is
        fast and has no dependence on CasADi expressions.
        """
        if self._backend == "pinocchio":
            full = self.base_config.copy()
            for i, idx in enumerate(self.active_indices):
                full[idx] = q_active_numeric[i]
            q = np.empty(int(self.pin_model.nq))
            q[0] = full[0]
            q[1] = full[1]
            q[2] = np.cos(full[2])
            q[3] = np.sin(full[2])
            q[4:] = full[3:]
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacement(
                self.pin_model,
                self.pin_data,
                self.pin_model.getFrameId(link_name),
            )
            pose = self.pin_data.oMf[self.pin_model.getFrameId(link_name)]
            M = np.eye(4)
            M[:3, :3] = pose.rotation
            M[:3, 3] = pose.translation
            return M

        # urdf2casadi fallback
        T = self._urdf2casadi_pose(link_name, self.q)
        T_fn = ca.Function("fk_eval", [self.q], [T.rotation, T.translation])
        rot_num, trans_num = T_fn(np.asarray(q_active_numeric, dtype=np.float64))
        M = np.eye(4)
        M[:3, :3] = np.asarray(rot_num, dtype=np.float64)
        M[:3, 3] = np.asarray(trans_num, dtype=np.float64).reshape(-1)
        return M

    def project(
        self,
        q_init: np.ndarray,
        residual: ca.SX,
        tol: float = 1e-8,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Project a joint configuration onto the manifold ``residual(q) = 0``.

        Runs damped Gauss-Newton on the CasADi Jacobian — the same
        iteration OMPL's ``ProjectedStateSpace`` runs internally — so
        the returned config will pass the planner's tolerance check
        and can be used directly as a start or goal state.
        """
        res_fn = ca.Function("proj_res", [self.q], [ca.reshape(residual, -1, 1)])
        jac_fn = ca.Function("proj_jac", [self.q], [ca.jacobian(residual, self.q)])
        q = np.asarray(q_init, dtype=np.float64).copy()
        for _ in range(max_iters):
            r = np.asarray(res_fn(q)).flatten()
            if np.linalg.norm(r) < tol:
                return q
            J = np.asarray(jac_fn(q))
            JJt = J @ J.T + 1e-10 * np.eye(J.shape[0])
            q -= J.T @ np.linalg.solve(JJt, r)
        raise RuntimeError(
            f"SymbolicContext.project failed to converge: "
            f"|residual|={np.linalg.norm(r):.2e} after {max_iters} iters"
        )


__all__ = ["SymbolicContext"]
