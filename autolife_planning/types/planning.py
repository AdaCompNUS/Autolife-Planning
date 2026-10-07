from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PlanningStatus(Enum):
    """Status of a motion planning attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"


@dataclass
class PlannerConfig:
    """Configuration parameters for the motion planner."""

    planner_name: str = "rrtc"
    time_limit: float = 10.0
    point_radius: float = 0.01
    simplify: bool = True
    interpolate: bool = True
    # Interpolation density: each edge of length d gets
    # ceil(d * resolution) segments; 0 uses OMPL's default
    # longest-valid-segment fraction.
    resolution: float = 64.0
    # Joint-range overrides by joint name, as (lower, upper) in metres
    # for the virtual base joints and radians otherwise.  Joints left
    # out keep the range compiled into the robot model (the base spans
    # ±10 m); the planner samples only inside these ranges.
    joint_limits: dict[str, tuple[float, float]] | None = None

    def __post_init__(self):
        valid_planners = (
            # RRT family
            "rrtc",
            "rrt",
            "rrtstar",
            "informed_rrtstar",
            "rrtsharp",
            "rrtxstatic",
            "strrtstar",
            "lbtrrt",
            "trrt",
            "bitrrt",
            # Informed trees (asymptotically optimal)
            "bitstar",
            "abitstar",
            "aitstar",
            "eitstar",
            "blitstar",
            # FMT
            "fmt",
            "bfmt",
            # KPIECE
            "kpiece",
            "bkpiece",
            "lbkpiece",
            # PRM family
            "prm",
            "prmstar",
            "lazyprm",
            "lazyprmstar",
            "spars",
            "spars2",
            # Exploration-based
            "est",
            "biest",
            "sbl",
            "stride",
            "pdst",
        )
        if self.planner_name not in valid_planners:
            raise ValueError(
                f"Unknown planner '{self.planner_name}'. "
                f"Supported: {', '.join(valid_planners)}"
            )
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0")
        if self.point_radius <= 0:
            raise ValueError("point_radius must be > 0")
        if self.resolution < 0:
            raise ValueError("resolution must be >= 0")
        if self.joint_limits is not None:
            limits = {}
            for name, bounds in self.joint_limits.items():
                try:
                    lower, upper = (float(b) for b in bounds)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"joint_limits[{name!r}] must be a (lower, upper) pair"
                    ) from None
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    raise ValueError(f"joint_limits[{name!r}] must be finite")
                if lower >= upper:
                    raise ValueError(
                        f"joint_limits[{name!r}] needs lower < upper, "
                        f"got ({lower}, {upper})"
                    )
                limits[name] = (lower, upper)
            self.joint_limits = limits


@dataclass
class PlanningResult:
    """Result of a motion planning attempt."""

    status: PlanningStatus
    path: np.ndarray | None
    planning_time_ns: int
    path_cost: float

    @property
    def success(self) -> bool:
        return self.status == PlanningStatus.SUCCESS
