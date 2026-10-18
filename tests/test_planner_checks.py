"""Checks of the low-level ``OmplVampPlanner``.

Most use the base subgroup (x, y, theta), the rest of the body frozen
at home, among sphere obstacles.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")

START = np.array([-1.5, 0.0, 0.0])
GOAL = np.array([1.5, 0.0, 0.0])


def base_planner():
    """Base (x, y, theta) planner, the rest of the body frozen at home."""
    from autolife_planning._ompl_vamp import OmplVampPlanner
    from autolife_planning.autolife import HOME_JOINTS

    planner = OmplVampPlanner([0, 1, 2], HOME_JOINTS.tolist())
    lower, upper = OmplVampPlanner.default_joint_limits()
    lower[:2], upper[:2] = -3.0, 3.0
    planner.set_joint_limits(lower, upper)
    return planner


def test_base_subgroup_checks_the_frozen_body():
    # A small sphere at head height: only the frozen upper body reaches it.
    planner = base_planner()
    planner.add_sphere([2.0, 0.0, 1.59], 0.05)
    assert not planner.validate(np.array([2.0, 0.0, 0.0]))
    assert planner.validate(np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("name", ["pdst", "kpiece", "bkpiece", "lbkpiece", "stride"])
def test_partial_motions_stop_at_obstacles(name):
    """These planners grow from the last valid state of a blocked motion."""
    planner = base_planner()
    # A wall across the straight line from START to GOAL, |y| <= 1.6.
    for y in np.arange(-1.5, 1.51, 0.1):
        planner.add_sphere([0.0, y, 0.3], 0.1)
    for _ in range(5):
        result = planner.plan(START, GOAL, name, 5.0, False, False, 0.0)
        assert result.solved
        # The base centre crosses x = 0 only beyond the ends of the wall.
        a, b = result.path[:-1], result.path[1:]
        cross = np.flatnonzero(np.sign(a[:, 0]) != np.sign(b[:, 0]))
        y = a[cross, 1] + (b[cross, 1] - a[cross, 1]) * a[cross, 0] / (
            a[cross, 0] - b[cross, 0]
        )
        assert np.all(np.abs(y) > 1.5)
