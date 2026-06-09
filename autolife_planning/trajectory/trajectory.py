"""Trajectory value type — a time-parameterised configuration stream.

Thin Python veneer over a backend trajectory handle: it keeps the handle
alive for continuous sampling at arbitrary times, and exposes convenience
accessors (``duration``, array-returning ``sample`` / ``sample_uniform``)
that return plain NumPy arrays so downstream consumers never touch backend
implementation types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """A time-optimal trajectory produced by
    :class:`~autolife_planning.trajectory.TimeOptimalParameterizer`.

    Instances are immutable handles around a concrete timing backend;
    query them via :meth:`position`, :meth:`velocity`,
    :meth:`acceleration`, or one of the batch samplers.
    """

    _handle: Any

    @property
    def duration(self) -> float:
        """Trajectory duration in seconds."""
        return float(self._handle.duration)

    # ── Pointwise sampling ────────────────────────────────────────────

    def position(self, t: float) -> np.ndarray:
        """Configuration at time ``t`` (seconds), clamped to ``[0, duration]``."""
        return np.asarray(self._handle.position(float(t)), dtype=np.float64)

    def velocity(self, t: float) -> np.ndarray:
        """Joint velocity at time ``t`` (seconds)."""
        return np.asarray(self._handle.velocity(float(t)), dtype=np.float64)

    def acceleration(self, t: float) -> np.ndarray:
        """Joint acceleration at time ``t`` (seconds)."""
        return np.asarray(self._handle.acceleration(float(t)), dtype=np.float64)

    # ── Batch sampling ────────────────────────────────────────────────

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample at the user-supplied ``times`` grid.

        Args:
            times: ``(T,)`` array of sample times in seconds.

        Returns:
            ``(positions, velocities, accelerations)`` — each ``(T, ndof)``.
        """
        times = np.ascontiguousarray(times, dtype=np.float64).reshape(-1)
        positions, velocities, accelerations = self._handle.sample(times)
        return (
            np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64),
            np.asarray(accelerations, dtype=np.float64),
        )

    def sample_uniform(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly-spaced rollout at step ``dt``.

        The returned ``times`` always start at ``0`` and end at
        :attr:`duration`, which may make the final step shorter than
        ``dt`` — this matches what a streaming controller expects.

        Args:
            dt: Sample interval in seconds (must be ``> 0``).

        Returns:
            ``(times, positions, velocities, accelerations)`` —
            ``times`` has shape ``(T,)``; state arrays have shape
            ``(T, ndof)``.
        """
        times, positions, velocities, accelerations = self._handle.sample_uniform(
            float(dt)
        )
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64),
            np.asarray(accelerations, dtype=np.float64),
        )
