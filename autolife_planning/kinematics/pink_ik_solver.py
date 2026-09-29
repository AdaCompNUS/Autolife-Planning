"""Constrained IK solver using Pink (differential IK on Pinocchio).

Provides singularity-robust IK with optional self-collision avoidance,
obstacle collision avoidance, and center-of-mass stability.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from autolife_planning.kinematics.collision_model import CollisionContext
from autolife_planning.kinematics.ik_solver_base import IKSolverBase
from autolife_planning.types import (
    ChainConfig,
    ConstrainedIKResult,
    IKResult,
    IKStatus,
    PinkIKConfig,
    SE3Pose,
)

pin = importlib.import_module("pinocchio")


def _get_chain_joint_ids(model: Any, base_frame: str, ee_frame: str) -> list[int]:
    """Return joint IDs on the kinematic chain from *base_frame* to *ee_frame*."""
    ee_fid = model.getFrameId(ee_frame)
    base_fid = model.getFrameId(base_frame)

    def _frame_parent_joint(frame: Any) -> int:
        # Pinocchio 3.x exposes Frame.parentJoint, while 2.x exposes Frame.parent.
        if hasattr(frame, "parentJoint"):
            return int(frame.parentJoint)
        if hasattr(frame, "parent"):
            return int(frame.parent)
        raise AttributeError(
            "Unsupported pinocchio Frame API: expected parentJoint (3.x) "
            "or parent (2.x)."
        )

    ee_joint = _frame_parent_joint(model.frames[ee_fid])
    base_joint = _frame_parent_joint(model.frames[base_fid])

    joint_ids: list[int] = []
    current = ee_joint
    while current != base_joint and current > 0:
        joint_ids.append(current)
        current = model.parents[current]

    joint_ids.reverse()
    return joint_ids


class PinkIKSolver(IKSolverBase):
    """Constrained IK solver built on Pink's QP-based differential IK.

    Supports Levenberg-Marquardt damping (singularity avoidance), posture
    regularization, CoM stability, and collision barriers.

    Use via the unified factory::

        solver = create_ik_solver("whole_body", side="left", backend="pink",
                                  config=PinkIKConfig(com_cost=0.1))
    """

    def __init__(
        self,
        chain_config: ChainConfig,
        config: PinkIKConfig | None = None,
        joint_names: list[str] | None = None,
        collision_context: CollisionContext | None = None,
    ) -> None:
        self._chain_config = chain_config
        self._config = config if isinstance(config, PinkIKConfig) else PinkIKConfig()

        # Reuse the collision context's model when available so that the
        # kinematic model and collision model share the same instance.
        if collision_context is not None:
            self._model = collision_context.model
            self._data = collision_context.data
        else:
            self._model = pin.buildModelFromUrdf(chain_config.urdf_path)
            self._data = self._model.createData()

        # Resolve EE frame
        if not self._model.existFrame(chain_config.ee_link):
            available = [
                self._model.frames[i].name for i in range(int(self._model.nframes))
            ]
            raise ValueError(
                f"Frame '{chain_config.ee_link}' not found. " f"Available: {available}"
            )
        self._ee_frame_id = self._model.getFrameId(chain_config.ee_link)

        # Determine controlled joints
        if joint_names is not None:
            model_names = list(self._model.names)
            self._joint_ids: list[int] = []
            for name in joint_names:
                if name not in model_names:
                    raise ValueError(f"Joint '{name}' not in model")
                self._joint_ids.append(self._model.getJointId(name))
            self._joint_names = list(joint_names)
        else:
            self._joint_ids = _get_chain_joint_ids(
                self._model, chain_config.base_link, chain_config.ee_link
            )
            self._joint_names = [str(self._model.names[jid]) for jid in self._joint_ids]

        # Pre-compute velocity-index set for controlled joints
        self._controlled_idx_v: set[int] = set()
        for jid in self._joint_ids:
            idx_v = self._model.joints[jid].idx_v
            nv = self._model.joints[jid].nv
            for k in range(nv):
                self._controlled_idx_v.add(idx_v + k)

        self._collision_context = collision_context

    # --- IKSolverBase properties ---

    @property
    def base_frame(self) -> str:
        return self._chain_config.base_link

    @property
    def ee_frame(self) -> str:
        return self._chain_config.ee_link

    @property
    def num_joints(self) -> int:
        return len(self._joint_ids)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    # --- Public helpers ---

    def set_collision_context(self, context: CollisionContext | None) -> None:
        """Replace the collision context (e.g. after updating obstacles)."""
        self._collision_context = context

    # --- IKSolverBase methods ---

    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        """Compute forward kinematics for the end effector."""
        q = self._to_full_q(joint_positions)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._ee_frame_id]
        return SE3Pose(
            position=np.array(oMf.translation, dtype=np.float64),
            rotation=np.array(oMf.rotation, dtype=np.float64),
        )

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: PinkIKConfig | None = None,
    ) -> IKResult:
        """Solve IK, returning a standard :class:`IKResult`.

        Accepts :class:`PinkIKConfig`; falls back to constructor default.
        """
        pink_config = config if isinstance(config, PinkIKConfig) else None
        result = self.solve_constrained(target_pose, seed, pink_config)
        return IKResult(
            status=result.status,
            joint_positions=result.joint_positions,
            final_error=result.final_error,
            iterations=result.iterations,
            position_error=result.position_error,
            orientation_error=result.orientation_error,
        )

    # --- Extended constrained solve ---

    def solve_constrained(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: PinkIKConfig | None = None,
    ) -> ConstrainedIKResult:
        """Full constrained IK solve with trajectory output.

        Input:
            target_pose: Desired end-effector pose.
            seed: Initial joint configuration (controlled joints).
                Uses Pinocchio neutral config if None.
            config: Override the default PinkIKConfig for this call.
        Output:
            ConstrainedIKResult including the iteration trajectory.
        """
        import pink
        from pink.limits import ConfigurationLimit
        from pink.tasks import ComTask, FrameTask, PostureTask

        cfg = config or self._config

        # --- Initial full configuration ---
        if seed is not None:
            q_init = self._to_full_q(np.asarray(seed, dtype=np.float64))
        else:
            q_init = pin.neutral(self._model)

        # --- Pink Configuration ---
        ck = self._collision_context
        if ck is not None and cfg.self_collision:
            configuration = pink.Configuration(
                self._model,
                self._data,
                q_init,
                collision_model=ck.collision_model,
                collision_data=ck.collision_data,
            )
        else:
            configuration = pink.Configuration(self._model, self._data, q_init)

        # --- Target ---
        target_se3 = pin.SE3(
            np.asarray(target_pose.rotation, dtype=np.float64),
            np.asarray(target_pose.position, dtype=np.float64),
        )

        # --- Tasks ---
        ee_task = FrameTask(
            self._chain_config.ee_link,
            position_cost=cfg.position_cost,
            orientation_cost=cfg.orientation_cost,
            lm_damping=cfg.lm_damping,
        )
        ee_task.set_target(target_se3)

        posture_task = PostureTask(cost=cfg.posture_cost)
        posture_task.set_target(q_init)

        tasks: list = [ee_task, posture_task]

        if cfg.com_cost > 0:
            com_task = ComTask(cost=cfg.com_cost)
            com_task.set_target_from_configuration(configuration)
            tasks.append(com_task)

        if cfg.camera_frame and cfg.camera_cost > 0:
            if not self._model.existFrame(cfg.camera_frame):
                raise ValueError(
                    f"Camera frame '{cfg.camera_frame}' not found in model"
                )
            camera_task = FrameTask(
                cfg.camera_frame,
                position_cost=cfg.camera_cost,
                orientation_cost=cfg.camera_cost,
                lm_damping=cfg.lm_damping,
            )
            camera_task.set_target_from_configuration(configuration)
            tasks.append(camera_task)

        # --- Limits ---
        limits = [ConfigurationLimit(self._model)]

        # --- Barriers ---
        barriers: list = []
        if cfg.self_collision and ck is not None:
            from pink.barriers import SelfCollisionBarrier

            n_pairs = len(ck.collision_model.collisionPairs)
            if n_pairs > 0:
                barriers.append(
                    SelfCollisionBarrier(
                        n_collision_pairs=min(cfg.collision_pairs, n_pairs),
                        gain=cfg.collision_gain,
                        d_min=cfg.collision_d_min,
                    )
                )

        # --- Iterative solve ---
        trajectory = [self._from_full_q(q_init)]

        for iteration in range(cfg.max_iterations):
            solver_name = cfg.solver
            velocity = None
            try:
                velocity = pink.solve_ik(
                    configuration,
                    tasks,
                    cfg.dt,
                    solver=solver_name,
                    limits=limits,
                    barriers=barriers or None,
                )
            except Exception as exc:
                # Some environments (e.g. py38 with qpsolvers extras) may not
                # have the requested solver backend. Fall back to osqp when the
                # configured solver is unavailable.
                msg = str(exc)
                solver_not_found = (
                    "does not seem to be installed" in msg
                    or exc.__class__.__name__ == "SolverNotFound"
                )
                if solver_name == "proxqp" and solver_not_found:
                    try:
                        velocity = pink.solve_ik(
                            configuration,
                            tasks,
                            cfg.dt,
                            solver="osqp",
                            limits=limits,
                            barriers=barriers or None,
                        )
                    except Exception as exc2:
                        exc = exc2
                    else:
                        # Fallback succeeded, continue the solve loop.
                        pass
                else:
                    # Keep ``exc`` for the failure path below.
                    pass

                # If fallback succeeded, ``velocity`` exists in local scope.
                if velocity is not None:
                    pass
                elif getattr(pink, "PinkError", None) is None or isinstance(
                    exc, getattr(pink, "PinkError")
                ):
                    # QP infeasible at this step (e.g. conflicting barriers) —
                    # return best-effort result so far.
                    q_current = configuration.q
                    pos_err, ori_err = self._compute_errors(q_current, target_pose)
                    return ConstrainedIKResult(
                        status=IKStatus.FAILED,
                        joint_positions=self._from_full_q(q_current),
                        final_error=pos_err + ori_err,
                        iterations=iteration,
                        position_error=pos_err,
                        orientation_error=ori_err,
                        trajectory=np.array(trajectory),
                    )
                else:
                    # Unknown backend exception style (pink/qpsolvers version
                    # mismatch). Return best-effort result instead of crashing.
                    q_current = configuration.q
                    pos_err, ori_err = self._compute_errors(q_current, target_pose)
                    return ConstrainedIKResult(
                        status=IKStatus.FAILED,
                        joint_positions=self._from_full_q(q_current),
                        final_error=pos_err + ori_err,
                        iterations=iteration,
                        position_error=pos_err,
                        orientation_error=ori_err,
                        trajectory=np.array(trajectory),
                    )

            # Zero velocity for uncontrolled joints
            for i in range(self._model.nv):
                if i not in self._controlled_idx_v:
                    velocity[i] = 0.0

            configuration.integrate_inplace(velocity, cfg.dt)

            q_current = configuration.q
            trajectory.append(self._from_full_q(q_current))

            # Convergence check
            pos_err, ori_err = self._compute_errors(q_current, target_pose)
            if pos_err < cfg.convergence_thresh and ori_err < cfg.orientation_thresh:
                return ConstrainedIKResult(
                    status=IKStatus.SUCCESS,
                    joint_positions=trajectory[-1],
                    final_error=pos_err + ori_err,
                    iterations=iteration + 1,
                    position_error=pos_err,
                    orientation_error=ori_err,
                    trajectory=np.array(trajectory),
                )

        # Did not converge — return best effort
        q_final = configuration.q
        pos_err, ori_err = self._compute_errors(q_final, target_pose)
        return ConstrainedIKResult(
            status=IKStatus.MAX_ITERATIONS,
            joint_positions=trajectory[-1],
            final_error=pos_err + ori_err,
            iterations=cfg.max_iterations,
            position_error=pos_err,
            orientation_error=ori_err,
            trajectory=np.array(trajectory),
        )

    # --- Internal helpers ---

    def _to_full_q(self, joint_positions: np.ndarray) -> np.ndarray:
        """Expand controlled joint values into a full Pinocchio config vector."""
        q = pin.neutral(self._model)
        for i, jid in enumerate(self._joint_ids):
            idx_q = self._model.joints[jid].idx_q
            q[idx_q] = joint_positions[i]
        return q

    def _from_full_q(self, q: np.ndarray) -> np.ndarray:
        """Extract controlled joint values from a full Pinocchio config vector."""
        return np.array(
            [q[self._model.joints[jid].idx_q] for jid in self._joint_ids],
            dtype=np.float64,
        )

    def _compute_errors(self, q: np.ndarray, target: SE3Pose) -> tuple[float, float]:
        """Return (position_error, orientation_error) for a full config."""
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        oMf = self._data.oMf[self._ee_frame_id]
        pos_err = float(np.linalg.norm(oMf.translation - target.position))
        R_err = np.asarray(oMf.rotation).T @ target.rotation
        ori_err = float(np.linalg.norm(Rotation.from_matrix(R_err).as_rotvec()))
        return pos_err, ori_err
