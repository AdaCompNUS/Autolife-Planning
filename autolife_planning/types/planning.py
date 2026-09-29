from __future__ import annotations

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
    # Interpolation density knobs (at most one may be nonzero):
    #   interpolate_count > 0 : total waypoint count (distance-weighted
    #       distribution across edges).
    #   resolution > 0.0      : waypoints per unit state-space distance
    #       — each edge of length d gets ceil(d * resolution) segments.
    #   both 0                : OMPL default longest-valid-segment
    #       fraction.
    interpolate_count: int = 0
    resolution: float = 64.0

    # Backward-compat mapping from old VAMP planner names
    _COMPAT_MAP: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        compat = {"fcit": "rrtstar", "aorrtc": "bitstar"}
        if self.planner_name in compat:
            import warnings

            new = compat[self.planner_name]
            warnings.warn(
                f"Planner '{self.planner_name}' is deprecated, "
                f"using '{new}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.planner_name = new

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
        if self.interpolate_count < 0:
            raise ValueError("interpolate_count must be >= 0")
        if self.resolution < 0:
            raise ValueError("resolution must be >= 0")
        if self.interpolate_count > 0 and self.resolution > 0:
            raise ValueError(
                "Specify at most one of interpolate_count (>0) or "
                "resolution (>0), not both."
            )


@dataclass
class PlanningResult:
    """Result of a motion planning attempt."""

    status: PlanningStatus
    path: np.ndarray | None
    planning_time_ns: int
    iterations: int
    path_cost: float

    @property
    def success(self) -> bool:
        return self.status == PlanningStatus.SUCCESS
