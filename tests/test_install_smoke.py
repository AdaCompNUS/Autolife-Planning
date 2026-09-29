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
- The default TOPP-RA time-parameterization backend runs.
- The FK backend that ``SymbolicContext`` relies on is usable (pinocchio
  OR urdf2casadi).
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


def test_default_time_parameterizer_runs():
    """Default TOPP-RA backend must parameterize a tiny path."""
    from autolife_planning.trajectory import TimeOptimalParameterizer

    path = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]])
    param = TimeOptimalParameterizer(
        max_velocity=np.ones(2),
        max_acceleration=np.ones(2) * 2.0,
    )
    traj = param.parameterize(path)
    assert traj.duration > 0.0


def test_trajectory_extension_loads_and_runs():
    """Native ``_time_parameterization`` must still support the TOTG method."""
    pytest.importorskip("autolife_planning._time_parameterization")
    from autolife_planning.trajectory import TimeOptimalParameterizer

    path = np.array([[0.0, 0.0], [0.5, 0.3], [1.0, 0.6]])
    param = TimeOptimalParameterizer(
        max_velocity=np.ones(2),
        max_acceleration=np.ones(2) * 2.0,
        method="totg",
    )
    traj = param.parameterize(path)
    assert traj.duration > 0.0


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


def test_symbolic_context_backend_available():
    """Either pinocchio.casadi or urdf2casadi must be importable post-install.

    This is the guardrail: if both backends fail to load, every Constraint
    / Cost authored on top of ``SymbolicContext`` breaks.  We surface that
    as a hard fail with a pip install hint rather than letting it silently
    bite the first user who tries the examples.

    When the failure happens, we re-import the candidate backends here so
    the actual error message reaches the CI log — ``symbolic.py`` swallows
    the import exceptions to keep the optional-dep contract.
    """
    import autolife_planning.planning.symbolic as sym

    if sym.pin is None and sym.URDFparser is None:
        # symbolic.py's blanket ``except Exception`` hid the real cause;
        # re-run each import here so the traceback shows up in CI.
        errors = {}
        for name in ("pinocchio", "pinocchio.casadi", "urdf2casadi"):
            try:
                __import__(name)
                errors[name] = "(import succeeded — symbolic.py state stale?)"
            except Exception as exc:  # noqa: BLE001 - we want the full reason
                errors[name] = f"{type(exc).__name__}: {exc}"
        pytest.fail(
            "No FK backend usable from SymbolicContext. Re-import results:\n"
            + "\n".join(f"  {name}: {msg}" for name, msg in errors.items())
            + "\n\nFix: ``pip install pin`` (preferred) or "
            "``pip install urdf2casadi`` (fallback)."
        )

    from autolife_planning.planning import SymbolicContext

    ctx = SymbolicContext("autolife_left_arm")
    assert len(ctx.active_indices) == 7
    # Smoke the FK path: position of a known link must be a 3-vector.
    pos = ctx.link_translation("Link_Left_Wrist_Lower_to_Gripper")
    assert pos.shape == (3, 1)


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
