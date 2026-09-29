"""Post-install smoke tests — minimal sanity check after ``pip install``.

These tests exist so CI can diagnose "did the wheel build/install
correctly?" in under a second, before spending minutes on the full
correctness suite.  The pip-compat workflow (see
``.github/workflows/pip-compat.yml``) runs this module first in a
fresh venv on every supported Python version.

Covered:

- Every public sub-package imports cleanly (types, autolife, planning,
  trajectory, utils).
- Each native extension loads (``_ompl_vamp``, ``_time_parameterization``).
- ``SymbolicContext``'s CasADi FK matches pinocchio's numeric FK.
- A single end-to-end plan + time-parameterize round-trip succeeds.

Correctness-level checks (velocity/accel bounds, planner convergence,
IK residuals, etc.) live in the neighbouring ``test_*.py`` modules.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── imports ──────────────────────────────────────────────────────────


def test_top_level_package_imports():
    import autolife_planning  # noqa: F401
    from autolife_planning import autolife, planning, trajectory, types  # noqa: F401


def test_autolife_robot_config_populated():
    from autolife_planning.autolife import (
        HOME_JOINTS,
        PLANNING_SUBGROUPS,
        autolife_robot_config,
    )

    assert HOME_JOINTS.shape == (24,)
    assert len(PLANNING_SUBGROUPS) > 0
    # The RobotConfig fields added for centralized timing limits must be
    # populated so ``AutolifePlanner.time_parameterize`` has defaults.
    assert autolife_robot_config.max_velocity is not None
    assert autolife_robot_config.max_velocity.shape == (24,)
    assert autolife_robot_config.max_acceleration is not None
    assert autolife_robot_config.max_acceleration.shape == (24,)


# ── native extensions ────────────────────────────────────────────────


def test_trajectory_extension_loads_and_runs():
    """Native ``_time_parameterization`` (TOPP-RA) must time a tiny path."""
    from autolife_planning._time_parameterization import compute_trajectory

    path = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]])
    traj = compute_trajectory(path, np.ones(2), np.ones(2) * 2.0)
    assert traj is not None and traj.duration > 0.0


def test_planner_extension_loads():
    pytest.importorskip("autolife_planning._ompl_vamp")
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    )
    assert planner is not None


# ── symbolic FK backend ──────────────────────────────────────────────


def test_symbolic_fk_matches_pinocchio():
    """``SymbolicContext``'s CasADi FK agrees with pinocchio's numeric FK."""
    import pinocchio as pin

    from autolife_planning.autolife import autolife_robot_config
    from autolife_planning.planning import SymbolicContext

    ctx = SymbolicContext("autolife")
    model = pin.buildModelFromUrdf(
        autolife_robot_config.urdf_path, pin.JointModelPlanar()
    )
    data = model.createData()
    names = autolife_robot_config.joint_names
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 24)
        q_pin = np.zeros(model.nq)
        q_pin[:4] = [q[0], q[1], np.cos(q[2]), np.sin(q[2])]
        for name, value in zip(names[3:], q[3:]):
            q_pin[model.joints[model.getJointId(name)].idx_q] = value
        pin.framesForwardKinematics(model, data, q_pin)
        for link in ("Link_Left_Gripper", "Link_Right_Gripper", "Link_Head"):
            np.testing.assert_allclose(
                ctx.evaluate_link_pose(link, q),
                data.oMf[model.getFrameId(link)].homogeneous,
                atol=1e-9,
            )


# ── end-to-end ───────────────────────────────────────────────────────


def test_end_to_end_plan_and_parameterize():
    """One pass through VAMP/OMPL planning + TOPP-RA timing.

    If this test passes for a given (python-version, wheel) pair, the
    wheel is practically useful for the single-arm planning + time-
    parameterization workflow — the core of what most users do.
    """
    pytest.importorskip("autolife_planning._ompl_vamp")
    from autolife_planning.autolife import HOME_JOINTS
    from autolife_planning.planning import create_planner
    from autolife_planning.trajectory import TimeOptimalParameterizer
    from autolife_planning.types import PlannerConfig

    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )
    start = planner.extract_config(HOME_JOINTS)
    goal = start.copy()
    # Forearm joint (index 4 in the left-arm subgroup): HOME=0.04, bounds
    # ±2.97, so +0.5 is safely interior on every joint we touch.
    goal[4] += 0.5

    result = planner.plan(start, goal, time_limit=2.0)
    assert result.success and result.path is not None and result.path.shape[0] >= 2

    param = TimeOptimalParameterizer(np.full(7, 0.5), np.full(7, 0.6))
    traj = param.parameterize(result.path)
    assert traj.duration > 0.0

    # Uniform rollout must be self-consistent with the declared duration.
    times, positions, velocities, accelerations = traj.sample_uniform(0.02)
    assert positions.shape[1] == 7
    assert velocities.shape == positions.shape
    assert accelerations.shape == positions.shape
    assert abs(times[-1] - traj.duration) < 1e-6
