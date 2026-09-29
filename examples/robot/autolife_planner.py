from __future__ import annotations

from functools import lru_cache

import numpy as np

from autolife_planning.autolife import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    autolife_robot_config,
)
from autolife_planning.planning import create_planner
from autolife_planning.trajectory import TimeOptimalParameterizer
from autolife_planning.types import PlannerConfig

SUPPORTED_GROUPS = {
    "autolife_left_arm",
    "autolife_right_arm",
    "autolife_dual_arm",
    "autolife_torso_left_arm",
    "autolife_torso_right_arm",
    "autolife_torso_dual_arm",
    "autolife_leg_torso_dual_arm",
}

FULL_DOF = len(autolife_robot_config.joint_names)


def _active_indices(group: str) -> np.ndarray:
    if group not in PLANNING_SUBGROUPS:
        raise ValueError(f"Unknown subgroup: {group}")
    subgroup_joints = PLANNING_SUBGROUPS[group]["joints"]
    full_joint_names = autolife_robot_config.joint_names
    return np.asarray([full_joint_names.index(j) for j in subgroup_joints], dtype=int)


def _validate_24d(name: str, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (FULL_DOF,):
        raise ValueError(f"{name} must be shape ({FULL_DOF},), got {q.shape}")
    return q


def _frozen_joints_same(
    group: str, start: np.ndarray, goal: np.ndarray, tol: float
) -> bool:
    active = _active_indices(group)
    frozen = np.setdiff1d(np.arange(FULL_DOF), active)
    return np.allclose(start[frozen], goal[frozen], atol=tol, rtol=0.0)


@lru_cache(maxsize=1)
def _leg_torso_dual_arm_cost():
    import casadi as ca

    from autolife_planning.planning import Cost, SymbolicContext

    # Penalise the horizontal (XY) offset between Link_Waist_Pitch_to_Waist_Yaw
    # and Link_Ground_Vehicle — i.e. keep the waist directly above the base.
    # residual = [dx, dy]  →  cost = dx² + dy²
    ctx = SymbolicContext("autolife_leg_torso_dual_arm")
    waist_pos = ctx.link_translation("Link_Waist_Pitch_to_Waist_Yaw")
    ground_pos = ctx.link_translation("Link_Ground_Vehicle")
    residual = waist_pos[:2] - ground_pos[:2]
    return Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="leg_torso_dual_arm_balance_cost",
        weight=10.0,
    )


class AutolifePlanner:
    """Motion planner for the Autolife robot.

    Two MotionPlanners are created eagerly at construction and reused across
    all plan_to_joints calls — the C++ VAMP collision environment is never
    rebuilt.  Each call only calls set_subgroup() to switch active joints.

    Example::

        planner = AutolifePlanner()
        start = planner.default_start()
        goal  = start.copy()
        goal[7:14] = [0.3, -0.5, 0.3, 1.8, -0.2, -0.6, 0.5]
        path = planner.plan_to_joints("autolife_left_arm", start, goal)
    """

    def __init__(
        self,
        planner_name: str = "bitstar",
        time_limit: float = 2.0,
        tol: float = 1e-6,
    ) -> None:
        """
        Args:
            planner_name: OMPL planner name used for all calls.
            time_limit:   Default planning timeout in seconds (overridable per call).
            tol:          Equality tolerance used to detect frozen-joint mismatches.
        """
        self.planner_name = planner_name
        self.time_limit = time_limit
        self.tol = tol
        self.current_group = None
        self._planner = create_planner(
            "autolife_left_arm",
            config=PlannerConfig(planner_name=planner_name),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_start() -> np.ndarray:
        """Return the home full-body configuration (24-DOF)."""
        return HOME_JOINTS.copy()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def plan_to_joints(
        self,
        group: str,
        start: np.ndarray,
        goal: np.ndarray,
        planner_name: str | None = None,
        time_limit: float | None = None,
    ) -> np.ndarray | str | None:
        """Plan a collision-free path for *group* between two 24-DOF configs.

        Args:
            group:        One of SUPPORTED_GROUPS.
            start:        Full 24-DOF start vector.
            goal:         Full 24-DOF goal vector.
            planner_name: Ignored (planner algorithm is fixed at construction).
            time_limit:   Planning timeout in seconds; defaults to ``self.time_limit``.

        Returns:
            - ``"not same"`` if any frozen joint differs between start and goal.
            - ``np.ndarray`` of shape ``(N, 24)`` on success.
            - ``None`` when planning times out or fails.
        """
        if group not in SUPPORTED_GROUPS:
            raise ValueError(
                f"group must be one of {sorted(SUPPORTED_GROUPS)}, got {group!r}"
            )

        start = _validate_24d("start", start)
        goal = _validate_24d("goal", goal)

        if not _frozen_joints_same(group, start, goal, tol=self.tol):
            return "not same"

        time_limit = time_limit if time_limit is not None else self.time_limit

        self._planner.set_subgroup(group, base_config=start)

        if (
            group == "autolife_leg_torso_dual_arm"
            and self.current_group != "autolife_leg_torso_dual_arm"
        ):
            # Cost dim must match the subgroup — set after set_subgroup.
            self._planner.set_costs([_leg_torso_dual_arm_cost()])
        else:
            self._planner.clear_costs()

        self.current_group = group

        start_sub = self._planner.extract_config(start)
        goal_sub = self._planner.extract_config(goal)

        result = self._planner.plan(start_sub, goal_sub, time_limit=time_limit)
        if not result.success or result.path is None:
            return None
        return self._planner.embed_path(result.path, base_config=start)

    @staticmethod
    def time_parameterize(
        path: np.ndarray,
        v_max: float | None = None,
        a_max: float | None = None,
        dt: float = 0.02,
        vel_scale: float = 1.0,
        a_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a joint-space waypoint path to a time-parameterized trajectory.

        Wraps :class:`autolife_planning.trajectory.TimeOptimalParameterizer`,
        which uses TOPP-RA by default and still supports the legacy
        MoveIt-style TOTG backend.

        Per-joint limits are sourced from
        ``autolife_robot_config.max_velocity`` / ``max_acceleration`` by
        default.  Passing a scalar ``v_max``/``a_max`` overrides with a
        uniform value broadcast across every joint.

        Args:
            path:      ``(N, DOF)`` waypoints from the motion planner.
            v_max:     Scalar override for per-joint velocity limit (rad/s).
                       ``None`` uses ``autolife_robot_config.max_velocity``.
            a_max:     Scalar override for per-joint acceleration limit
                       (rad/s²). ``None`` uses
                       ``autolife_robot_config.max_acceleration``.
            dt:        Output sample period (seconds).
            vel_scale: Fractional scale applied to the velocity limit
                       (must be in (0, 1]).
            a_scale:   Fractional scale applied to the acceleration limit
                       (must be in (0, 1]).

        Returns:
            times: ``(M,)`` timestamps of the output trajectory.
            traj:  ``(M, DOF)`` joint positions sampled at those timestamps.
        """
        path = np.asarray(path, dtype=np.float64)
        N, DOF = path.shape
        if N == 1:
            return np.array([0.0]), path.copy()

        if v_max is None:
            if autolife_robot_config.max_velocity is None:
                raise ValueError(
                    "v_max not supplied and autolife_robot_config.max_velocity "
                    "is None — either pass a scalar v_max or populate the "
                    "robot config."
                )
            max_velocity = np.asarray(
                autolife_robot_config.max_velocity, dtype=np.float64
            )
        else:
            max_velocity = np.full(DOF, v_max, dtype=np.float64)

        if a_max is None:
            if autolife_robot_config.max_acceleration is None:
                raise ValueError(
                    "a_max not supplied and autolife_robot_config."
                    "max_acceleration is None — either pass a scalar a_max "
                    "or populate the robot config."
                )
            max_acceleration = np.asarray(
                autolife_robot_config.max_acceleration, dtype=np.float64
            )
        else:
            max_acceleration = np.full(DOF, a_max, dtype=np.float64)

        if max_velocity.shape != (DOF,) or max_acceleration.shape != (DOF,):
            raise ValueError(
                f"limit shape mismatch: path DOF={DOF}, "
                f"max_velocity={max_velocity.shape}, "
                f"max_acceleration={max_acceleration.shape}"
            )

        param = TimeOptimalParameterizer(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        traj = param.parameterize(
            path,
            velocity_scaling=vel_scale,
            acceleration_scaling=a_scale,
        )
        times, positions, _, _ = traj.sample_uniform(dt)
        return times, positions
