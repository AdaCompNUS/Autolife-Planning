"""Time-optimal path parameterization (TOPP-RA) — Python front-end.

Wraps the C++ ``_time_parameterization`` extension with a reusable
parameteriser object and a convenience one-shot function.  The
algorithm is Pham & Pham (2018) "A New Approach to Time-Optimal Path
Parameterization Based on Reachability Analysis", vendored from the C++
core of `toppra <https://github.com/hungpham2511/toppra>`_ — see
``ext/time_parameterization/toppra/LICENSE`` for attribution.
"""

from __future__ import annotations

import numpy as np

from .trajectory import Trajectory


class TimeOptimalParameterizer:
    """Time-optimal path parameteriser with fixed joint limits.

    The piecewise-linear path is resampled every ``knot_spacing`` and
    joined by a cubic spline, and TOPP-RA finds the fastest velocity
    profile along it that starts and ends at rest.  The spline passes
    through every waypoint and deviates from the straight segments by
    about ``knot_spacing / 10``, rounding the corners.

    Args:
        max_velocity: ``(ndof,)`` per-joint velocity bound.  Units must
            match the path (rad/s for revolute joints, m/s for prismatic).
        max_acceleration: ``(ndof,)`` per-joint acceleration bound.
        knot_spacing: Spline knot spacing along the path (path units).
            Smaller follows the corners more tightly, at the cost of
            slower cornering and more computation.
    """

    def __init__(
        self,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        knot_spacing: float = 0.1,
    ):
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64).reshape(-1)
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64).reshape(
            -1
        )
        self.knot_spacing = float(knot_spacing)

    @property
    def num_dof(self) -> int:
        return int(self.max_velocity.shape[0])

    def parameterize(
        self,
        path: np.ndarray,
        velocity_scaling: float = 1.0,
        acceleration_scaling: float = 1.0,
    ) -> Trajectory:
        """Time-parameterize an ``(N, ndof)`` joint-space path.

        Args:
            path: ``(N, ndof)`` waypoints, at least two distinct.
            velocity_scaling: Factor in ``(0, 1]`` on the velocity limit.
            acceleration_scaling: Factor in ``(0, 1]`` on the
                acceleration limit.

        Raises:
            ValueError: If TOPP-RA finds no feasible parameterization.
        """
        from autolife_planning._time_parameterization import compute_trajectory

        handle = compute_trajectory(
            np.asarray(path, dtype=np.float64),
            self.max_velocity * velocity_scaling,
            self.max_acceleration * acceleration_scaling,
            self.knot_spacing,
        )
        if handle is None:
            raise ValueError("TOPP-RA found no feasible parameterization")
        return Trajectory(handle)


def parameterize_path(
    path: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    knot_spacing: float = 0.1,
) -> Trajectory:
    """One-shot :meth:`TimeOptimalParameterizer.parameterize`."""
    return TimeOptimalParameterizer(
        max_velocity, max_acceleration, knot_spacing
    ).parameterize(path)
