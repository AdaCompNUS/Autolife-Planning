"""TOPP-RA time-parameterization backend.

The public trajectory API is intentionally shared with the C++ TOTG backend:
this module returns a small handle with ``duration``, point sampling, and batch
sampling methods so :class:`autolife_planning.trajectory.Trajectory` can wrap
either implementation without callers changing code.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


TOPPRA_BC_TYPE = "natural"
TOPPRA_PARAMETRIZER = "ParametrizeConstAccel"
TOPPRA_SOLVER = "seidel"
TOPPRA_MIN_GRIDPOINTS = 100
TOPPRA_GRIDPOINT_ERROR = 1e-4


class ToppraTrajectoryHandle:
    """Duck-typed trajectory handle backed by ``toppra`` output."""

    def __init__(self, trajectory: Any) -> None:
        self._trajectory = trajectory
        interval = np.asarray(trajectory.path_interval, dtype=np.float64)
        self._start = float(interval[0])
        self._end = float(interval[1])
        self._duration = self._end - self._start
        if self._duration <= 0:
            raise ValueError("TOPP-RA produced a non-positive duration")
        self._num_dof = int(np.asarray(trajectory(self._start)).reshape(-1).shape[0])

    @property
    def duration(self) -> float:
        return self._duration

    def position(self, t: float) -> np.ndarray:
        return self._eval(float(t), order=0)

    def velocity(self, t: float) -> np.ndarray:
        return self._eval(float(t), order=1)

    def acceleration(self, t: float) -> np.ndarray:
        return self._eval(float(t), order=2)

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.ascontiguousarray(times, dtype=np.float64).reshape(-1)
        if times.size == 0:
            empty = np.empty((0, self._num_dof), dtype=np.float64)
            return empty.copy(), empty.copy(), empty.copy()

        toppra_times = self._to_toppra_times(times)
        positions = np.asarray(self._trajectory(toppra_times, 0), dtype=np.float64)
        velocities = np.asarray(self._trajectory(toppra_times, 1), dtype=np.float64)
        accelerations = np.asarray(self._trajectory(toppra_times, 2), dtype=np.float64)
        return (
            positions.reshape(times.size, self._num_dof),
            velocities.reshape(times.size, self._num_dof),
            accelerations.reshape(times.size, self._num_dof),
        )

    def sample_uniform(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        times = np.arange(0.0, self._duration, float(dt), dtype=np.float64)
        if times.size == 0:
            times = np.array([0.0], dtype=np.float64)
        if abs(times[-1] - self._duration) > 1e-12:
            times = np.append(times, self._duration)
        else:
            times[-1] = self._duration

        positions, velocities, accelerations = self.sample(times)
        return times, positions, velocities, accelerations

    def _eval(self, t: float, order: int) -> np.ndarray:
        toppra_t = float(self._to_toppra_times(np.array([t], dtype=np.float64))[0])
        return np.asarray(self._trajectory(toppra_t, order), dtype=np.float64).reshape(
            self._num_dof
        )

    def _to_toppra_times(self, times: np.ndarray) -> np.ndarray:
        return np.clip(times, 0.0, self._duration) + self._start


def compute_toppra_trajectory(
    waypoints: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
) -> ToppraTrajectoryHandle | None:
    """Compute a TOPP-RA trajectory through joint-space waypoints."""

    _prepare_matplotlib_cache()

    try:
        import toppra as ta
        import toppra.algorithm as toppra_algorithm
        import toppra.constraint as toppra_constraint
    except ImportError as exc:  # pragma: no cover - exercised in packaging.
        raise ImportError(
            "TOPP-RA time parameterization requires the 'toppra' package. "
            "Install autolife_planning with its runtime dependencies."
        ) from exc

    path_positions = _chord_length_positions(waypoints)
    geometric_path = ta.SplineInterpolator(
        path_positions,
        waypoints,
        bc_type=TOPPRA_BC_TYPE,
    )
    constraints = [
        toppra_constraint.JointVelocityConstraint(max_velocity),
        toppra_constraint.JointAccelerationConstraint(max_acceleration),
    ]

    try:
        instance = toppra_algorithm.TOPPRA(
            constraints,
            geometric_path,
            solver_wrapper=TOPPRA_SOLVER,
            parametrizer=TOPPRA_PARAMETRIZER,
            gridpt_max_err_threshold=TOPPRA_GRIDPOINT_ERROR,
            gridpt_min_nb_points=max(TOPPRA_MIN_GRIDPOINTS, waypoints.shape[0]),
        )
        trajectory = instance.compute_trajectory(0.0, 0.0)
    except Exception:
        return None

    if trajectory is None:
        return None
    return ToppraTrajectoryHandle(trajectory)


def _chord_length_positions(waypoints: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if positions[-1] <= 0.0:
        raise ValueError("path must contain at least two distinct waypoints")
    return positions


def _prepare_matplotlib_cache() -> None:
    """Avoid noisy matplotlib cache warnings from toppra imports."""

    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "autolife-planning-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
