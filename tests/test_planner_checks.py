"""Checks of the low-level ``OmplVampPlanner``.

Most use the base subgroup (x, y, theta), the rest of the body frozen
at home, among sphere obstacles.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")


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
