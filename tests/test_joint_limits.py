"""Configurable joint limits: overrides by name, kept across subgroups."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")

from autolife_planning.autolife import autolife_robot_config  # noqa: E402
from autolife_planning.planning import create_planner  # noqa: E402
from autolife_planning.types import PlannerConfig  # noqa: E402

NAMES = autolife_robot_config.joint_names
BASE_X = NAMES.index("Joint_Virtual_X")
WIDE_BASE = {"Joint_Virtual_X": (-1.0, 16.0), "Joint_Virtual_Y": (-1.0, 11.0)}


def compiled_limits():
    from autolife_planning._ompl_vamp import OmplVampPlanner

    return OmplVampPlanner.default_joint_limits()


def test_defaults_are_the_compiled_limits():
    lower, upper = compiled_limits()
    assert lower.shape == upper.shape == (len(NAMES),)
    assert np.all(lower < upper)
    assert (lower[BASE_X], upper[BASE_X]) == (-10.0, 10.0)

    planner = create_planner("autolife")
    np.testing.assert_array_equal(planner.bounds[0], lower)
    np.testing.assert_array_equal(planner.bounds[1], upper)
    assert planner.joint_limits["Joint_Virtual_X"] == (-10.0, 10.0)


def test_config_overrides_only_the_named_joints():
    planner = create_planner("autolife", config=PlannerConfig(joint_limits=WIDE_BASE))
    lower, upper = compiled_limits()
    expected_lower, expected_upper = lower.copy(), upper.copy()
    for name, (lo, hi) in WIDE_BASE.items():
        expected_lower[NAMES.index(name)] = lo
        expected_upper[NAMES.index(name)] = hi
    np.testing.assert_array_equal(planner.bounds[0], expected_lower)
    np.testing.assert_array_equal(planner.bounds[1], expected_upper)
    assert planner.joint_limits["Joint_Virtual_Y"] == (-1.0, 11.0)


def test_widened_base_reaches_beyond_the_compiled_range():
    config = PlannerConfig(planner_name="rrtc", time_limit=1.0)
    start, goal = np.zeros(3), np.array([12.0, 0.5, 0.0])

    blocked = create_planner("autolife_base", config=config)
    assert not blocked.plan(start, goal).success

    wide = create_planner(
        "autolife_base",
        config=PlannerConfig(
            planner_name="rrtc", time_limit=1.0, joint_limits=WIDE_BASE
        ),
    )
    result = wide.plan(start, goal)
    assert result.success
    np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)
    lo, hi = wide.bounds
    assert np.all(result.path >= lo - 1e-9) and np.all(result.path <= hi + 1e-9)


def test_narrowed_joint_bounds_sampling():
    name = "Joint_Left_Elbow"
    limits = {name: (0.5, 0.9)}
    planner = create_planner(
        "autolife_left_arm", config=PlannerConfig(joint_limits=limits)
    )
    column = planner.joint_names.index(name)
    assert planner.bounds[0][column] == 0.5 and planner.bounds[1][column] == 0.9
    np.random.seed(0)
    for _ in range(5):
        assert 0.5 <= planner.sample_valid()[column] <= 0.9


def test_limits_survive_subgroup_switches(home_joints):
    planner = create_planner("autolife", config=PlannerConfig(joint_limits=WIDE_BASE))
    planner.set_subgroup("autolife_base", base_config=home_joints)
    assert planner.bounds[1][0] == 16.0
    planner.set_subgroup("autolife_left_arm")
    planner.set_subgroup("autolife")
    assert planner.bounds[1][BASE_X] == 16.0


def test_set_joint_limits_replaces_and_none_restores():
    planner = create_planner("autolife", config=PlannerConfig(joint_limits=WIDE_BASE))
    planner.set_joint_limits({"Joint_Virtual_Y": (0.0, 5.0)})
    assert planner.joint_limits["Joint_Virtual_X"] == (-10.0, 10.0)
    assert planner.joint_limits["Joint_Virtual_Y"] == (0.0, 5.0)
    planner.set_joint_limits(None)
    lower, upper = compiled_limits()
    np.testing.assert_array_equal(planner.bounds[0], lower)
    np.testing.assert_array_equal(planner.bounds[1], upper)


@pytest.mark.parametrize(
    "limits",
    [
        {"Joint_Virtual_X": (5.0, 5.0)},
        {"Joint_Virtual_X": (6.0, 5.0)},
        {"Joint_Virtual_X": (float("nan"), 5.0)},
        {"Joint_Virtual_X": (0.0, float("inf"))},
        {"Joint_Virtual_X": (0.0,)},
        {"Joint_Virtual_X": "wide"},
    ],
)
def test_config_rejects_bad_bounds(limits):
    with pytest.raises(ValueError):
        PlannerConfig(joint_limits=limits)


def test_unknown_joint_is_rejected_and_limits_stay():
    planner = create_planner("autolife", config=PlannerConfig(joint_limits=WIDE_BASE))
    with pytest.raises(ValueError, match="Unknown joint"):
        planner.set_joint_limits({"Joint_Virtual_Z": (0.0, 1.0)})
    with pytest.raises(ValueError):
        planner.set_joint_limits({"Joint_Virtual_X": (3.0, 1.0)})
    assert planner.joint_limits["Joint_Virtual_X"] == (-1.0, 16.0)


def test_binding_validates_full_body_arrays():
    from autolife_planning._ompl_vamp import OmplVampPlanner

    planner = OmplVampPlanner()
    lower, upper = compiled_limits()
    with pytest.raises(ValueError):
        planner.set_joint_limits(lower[:3], upper[:3])
    with pytest.raises(ValueError):
        planner.set_joint_limits(upper, lower)
    np.testing.assert_array_equal(planner.joint_limits()[0], lower)
