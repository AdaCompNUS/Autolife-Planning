"""Time parameterization for joint-space paths.

Converts a piecewise-linear ``(N, ndof)`` waypoint path — as produced by
:class:`~autolife_planning.planning.MotionPlanner` — into an executable
:class:`Trajectory` with continuous velocity and bounded acceleration,
via TOPP-RA time-optimal path parameterization (Pham & Pham, 2018).

Typical use::

    from autolife_planning.trajectory import TimeOptimalParameterizer

    param = TimeOptimalParameterizer(vel_limits, acc_limits)
    traj = param.parameterize(path)                 # path: (N, ndof)
    times, pos, vel, acc = traj.sample_uniform(dt=0.01)
"""

from .toppra import TimeOptimalParameterizer, parameterize_path
from .trajectory import Trajectory

__all__ = [
    "Trajectory",
    "TimeOptimalParameterizer",
    "parameterize_path",
]
