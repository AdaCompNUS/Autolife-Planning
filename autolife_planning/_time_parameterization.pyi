"""Type stubs for the ``_time_parameterization`` C++ extension.

The actual implementation lives in ``ext/time_parameterization/`` and
ships as a compiled ``_time_parameterization.cpython-*.so`` next to this
file in the installed package.  Stubs mirror the nanobind bindings so
type checkers can resolve ``import autolife_planning._time_parameterization``.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

DEFAULT_PATH_TOLERANCE: Final[float]

class TotgTrajectory:
    """Opaque handle to a parameterised trajectory.

    Queries are C¹-consistent: :meth:`position` / :meth:`velocity` /
    :meth:`acceleration` sample a quadratic segment between the
    algorithm's internal forward-integration grid points.
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
    max_deviation: float = ...,
    time_step: float = 1e-3,
) -> TotgTrajectory | None:
    """Build a time-optimal trajectory for a piecewise-linear path.

    Returns ``None`` if ``waypoints`` has fewer than two rows, the
    velocity/acceleration bounds are infeasible, or the integrator
    failed (e.g. a 180-degree reversal in the waypoints).
    """
    ...
