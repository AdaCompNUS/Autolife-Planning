"""Pure-Python dataclass tests — no native extension required."""

from __future__ import annotations

import numpy as np
import pytest

from autolife_planning.types import (
    IKConfig,
    IKResult,
    IKStatus,
    PinkIKConfig,
    PlannerConfig,
    PlanningResult,
    PlanningStatus,
    SolveType,
)


class TestPlannerConfig:
    def test_defaults(self):
        cfg = PlannerConfig()
        assert cfg.planner_name == "rrtc"
        assert cfg.time_limit > 0
        assert cfg.simplify is True

    @pytest.mark.parametrize(
        "name", ["rrtc", "rrtstar", "bitstar", "aitstar", "prmstar", "kpiece"]
    )
    def test_known_planner_names_accepted(self, name: str):
        assert PlannerConfig(planner_name=name).planner_name == name

    def test_unknown_planner_rejected(self):
        with pytest.raises(ValueError, match="Unknown planner"):
            PlannerConfig(planner_name="not_a_real_planner")

    def test_legacy_names_rewritten(self):
        with pytest.warns(DeprecationWarning):
            cfg = PlannerConfig(planner_name="fcit")
        assert cfg.planner_name == "rrtstar"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time_limit": 0},
            {"point_radius": 0},
            {"interpolate_count": -1},
            {"resolution": -0.1},
            {"interpolate_count": 5, "resolution": 1.0},
        ],
    )
    def test_invalid_args_rejected(self, kwargs):
        with pytest.raises(ValueError):
            PlannerConfig(**kwargs)


class TestIKConfig:
    def test_defaults(self):
        cfg = IKConfig()
        assert cfg.solve_type == SolveType.SPEED
        assert cfg.timeout > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"timeout": 0},
            {"epsilon": 0},
            {"max_attempts": 0},
            {"position_tolerance": 0},
            {"orientation_tolerance": 0},
        ],
    )
    def test_invalid_args_rejected(self, kwargs):
        with pytest.raises(ValueError):
            IKConfig(**kwargs)


class TestPinkIKConfig:
    def test_defaults(self):
        cfg = PinkIKConfig()
        assert cfg.dt > 0
        assert cfg.solver
        assert cfg.coupled_joints == []

    def test_invalid_dt_rejected(self):
        with pytest.raises(ValueError):
            PinkIKConfig(dt=0)

    def test_invalid_collision_args_rejected(self):
        with pytest.raises(ValueError):
            PinkIKConfig(collision_pairs=0)
        with pytest.raises(ValueError):
            PinkIKConfig(collision_d_min=-0.01)


class TestResultDataclasses:
    def test_planning_result_success_property(self):
        ok = PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=np.zeros((2, 7)),
            planning_time_ns=0,
            iterations=0,
            path_cost=0.0,
        )
        bad = PlanningResult(
            status=PlanningStatus.FAILED,
            path=None,
            planning_time_ns=0,
            iterations=0,
            path_cost=float("inf"),
        )
        assert ok.success is True
        assert bad.success is False

    def test_ik_result_success_property(self):
        ok = IKResult(
            status=IKStatus.SUCCESS,
            joint_positions=np.zeros(7),
            final_error=0.0,
            iterations=1,
            position_error=0.0,
            orientation_error=0.0,
        )
        bad = IKResult(
            status=IKStatus.FAILED,
            joint_positions=None,
            final_error=1.0,
            iterations=10,
            position_error=1.0,
            orientation_error=1.0,
        )
        assert ok.success is True
        assert bad.success is False
