"""Type stubs for the ``_time_parameterization`` C++ extension.

The actual implementation lives in ``ext/time_parameterization/`` and
ships as a compiled ``_time_parameterization.cpython-*.so`` next to this
file in the installed package.  Stubs mirror the nanobind bindings so
type checkers can resolve ``import autolife_planning._time_parameterization``.
"""

import numpy as np
from numpy.typing import NDArray

class ToppraTrajectory:
    """Opaque handle to a parameterised trajectory.

    A cubic spline along the waypoint path, timed by TOPP-RA: the path
    velocity is piecewise linear in time between grid points (constant
    path acceleration), so joint velocity is continuous.
    """

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        ...

    def position(self, t: float) -> NDArray[np.float64]:
        """Configuration at time ``t`` (seconds)."""
        ...

    def velocity(self, t: float) -> NDArray[np.float64]:
        """Joint velocity at time ``t`` (seconds)."""
        ...

    def acceleration(self, t: float) -> NDArray[np.float64]:
        """Joint acceleration at time ``t`` (seconds)."""
        ...

    def sample(
        self,
        times: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Batched sampling at ``times`` — returns ``(T, ndof)`` matrices
        of (position, velocity, acceleration).
        """
        ...

    def sample_uniform(
        self,
        dt: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Uniform rollout with step ``dt``.

        Returns ``(times, positions, velocities, accelerations)``.
        ``times`` starts at ``0`` and ends at :attr:`duration`.
        """
        ...

def compute_trajectory(
    waypoints: NDArray[np.float64],
    max_velocity: NDArray[np.float64],
    max_acceleration: NDArray[np.float64],
    knot_spacing: float = 0.1,
) -> ToppraTrajectory | None:
    """Time-optimal trajectory along the piecewise-linear path
    ``waypoints`` ``(N, ndof)``, starting and ending at rest.

    The path is resampled every ``knot_spacing`` and splined; the result
    deviates from it by about ``knot_spacing / 10``.  Returns ``None`` if
    TOPP-RA finds no feasible parameterization; raises ``ValueError`` if the
    limits do not match the path's DOF or the path has no length.
    """
    ...
