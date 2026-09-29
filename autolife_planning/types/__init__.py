"""Portable, behaviour-free data shapes shared across the package.

Anything in here is a plain ``@dataclass`` or ``Enum`` whose ``__post_init__``
does at most argument validation.  No I/O, no JIT compilation, no
backend handles.  These are the types that cross module boundaries —
results returned by solvers, configs handed to factories, poses
exchanged between IK and planning.

A ``@dataclass`` that *carries behaviour* — for example
:class:`autolife_planning.planning.costs.Cost` and
:class:`autolife_planning.planning.constraints.Constraint`, whose
``__post_init__`` generates C code, compiles a ``.so``, and manages a
disk cache — does not belong here.  Such classes live next to their
behaviour, since splitting the data shape away from the only logic
that gives it meaning would be artificial.

Likewise, dataclasses whose fields are tied to a specific backend
(e.g. :class:`autolife_planning.kinematics.pinocchio_fk.PinocchioContext`,
whose ``model`` is a ``pinocchio.Model`` handle) live with that
backend rather than in this neutral package — a pure type module
should not depend on optional runtime libraries.
"""

from .geometry import SE3Pose
from .ik import (
    ConstrainedIKResult,
    CoupledJoint,
    IKConfig,
    IKResult,
    IKStatus,
    PinkIKConfig,
    SolveType,
)
from .planning import PlannerConfig, PlanningResult, PlanningStatus
from .robot import (
    CameraConfig,
    ChainConfig,
    RobotConfig,
)

__all__ = [
    # Geometry
    "SE3Pose",
    # IK
    "ConstrainedIKResult",
    "CoupledJoint",
    "IKConfig",
    "IKResult",
    "IKStatus",
    "PinkIKConfig",
    "SolveType",
    # Robot
    "CameraConfig",
    "ChainConfig",
    "RobotConfig",
    # Planning
    "PlannerConfig",
    "PlanningResult",
    "PlanningStatus",
]
