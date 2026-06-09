"""Tests for the time-parameterization module.

These exercise the public trajectory wrapper and its TOPP-RA/TOTG
backends.  No robot URDFs or motion planner are needed — we generate
simple synthetic paths in low-dimensional joint space.
"""

from __future__ import annotations

import numpy as np
import pytest

from autolife_planning.trajectory import (  # noqa: E402
    TimeOptimalParameterizer,
    Trajectory,
    parameterize_path,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _straight_path(ndof: int = 3, n_waypoints: int = 10) -> np.ndarray:
    """Straight-line path from origin to unit vector."""
    start = np.zeros(ndof)
    end = np.ones(ndof)
    return np.linspace(start, end, n_waypoints)


def _zigzag_path(ndof: int = 3) -> np.ndarray:
    """Multi-segment path with corners."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.3, -0.2],
            [1.0, 0.6, 0.1],
            [1.2, 0.4, 0.3],
        ]
    )[:, :ndof]


# ── Construction / validation ────────────────────────────────────────


class TestParameterizerConstruction:
    def test_basic_construction(self):
        param = TimeOptimalParameterizer(
            max_velocity=np.ones(3),
            max_acceleration=np.ones(3) * 2,
        )
        assert param.num_dof == 3
        assert param.method == "toppra"

    def test_accepts_totg_method(self):
        param = TimeOptimalParameterizer(
            max_velocity=np.ones(3),
            max_acceleration=np.ones(3) * 2,
            method="totg",
        )
        assert param.method == "totg"

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            TimeOptimalParameterizer(np.ones(3), np.ones(3), method="unknown")

    def test_rejects_zero_velocity(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TimeOptimalParameterizer(np.array([1.0, 0.0, 1.0]), np.ones(3))

    def test_rejects_negative_acceleration(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TimeOptimalParameterizer(np.ones(3), np.array([1.0, -1.0, 1.0]))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            TimeOptimalParameterizer(np.ones(3), np.ones(4))

    def test_rejects_negative_deviation(self):
        with pytest.raises(ValueError, match="strictly positive"):
            TimeOptimalParameterizer(np.ones(3), np.ones(3), max_deviation=-0.1)


# ── Parameterization ────────────────────────────────────────────────


class TestParameterize:
    @pytest.fixture()
    def param(self):
        return TimeOptimalParameterizer(
            max_velocity=np.ones(3),
            max_acceleration=np.ones(3) * 2,
        )

    def test_straight_line(self, param):
        path = _straight_path(3, 10)
        traj = param.parameterize(path)
        assert isinstance(traj, Trajectory)
        assert traj.duration > 0

    def test_zigzag(self, param):
        traj = param.parameterize(_zigzag_path(3))
        assert traj.duration > 0

    def test_two_waypoints(self, param):
        path = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        traj = param.parameterize(path)
        assert traj.duration > 0

    def test_rejects_single_waypoint(self, param):
        with pytest.raises(ValueError, match="at least 2"):
            param.parameterize(np.array([[0, 0, 0]], dtype=float))

    def test_rejects_wrong_dof(self, param):
        with pytest.raises(ValueError, match="DOF"):
            param.parameterize(np.array([[0, 0], [1, 1]], dtype=float))

    def test_rejects_bad_scaling(self, param):
        path = _straight_path(3, 5)
        with pytest.raises(ValueError, match="velocity_scaling"):
            param.parameterize(path, velocity_scaling=0.0)
        with pytest.raises(ValueError, match="acceleration_scaling"):
            param.parameterize(path, acceleration_scaling=1.5)

    def test_deduplicates_identical_waypoints(self, param):
        path = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        traj = param.parameterize(path)
        assert traj.duration > 0

    def test_totg_backend_remains_available(self):
        pytest.importorskip("autolife_planning._time_parameterization")
        param = TimeOptimalParameterizer(
            max_velocity=np.ones(3),
            max_acceleration=np.ones(3) * 2,
            method="totg",
        )
        traj = param.parameterize(_zigzag_path(3))
        assert traj.duration > 0


# ── Trajectory queries ───────────────────────────────────────────────


class TestTrajectoryQueries:
    @pytest.fixture()
    def traj(self):
        param = TimeOptimalParameterizer(np.ones(3), np.ones(3) * 2)
        return param.parameterize(_zigzag_path(3))

    def test_position_at_boundaries(self, traj):
        path = _zigzag_path(3)
        p0 = traj.position(0.0)
        pend = traj.position(traj.duration)
        np.testing.assert_allclose(p0, path[0], atol=1e-6)
        np.testing.assert_allclose(pend, path[-1], atol=1e-6)

    def test_velocity_at_start_and_end(self, traj):
        # Both supported backends start and end at rest. Residual is small
        # but nonzero due to numerical interpolation inside the backend.
        v0 = traj.velocity(0.0)
        vend = traj.velocity(traj.duration)
        np.testing.assert_allclose(v0, 0.0, atol=5e-3)
        np.testing.assert_allclose(vend, 0.0, atol=5e-3)

    def test_sample_returns_correct_shapes(self, traj):
        times = np.linspace(0, traj.duration, 50)
        p, v, a = traj.sample(times)
        assert p.shape == (50, 3)
        assert v.shape == (50, 3)
        assert a.shape == (50, 3)

    def test_sample_uniform(self, traj):
        t, p, v, a = traj.sample_uniform(dt=0.05)
        assert t[0] == 0.0
        assert abs(t[-1] - traj.duration) < 1e-10
        assert p.shape[0] == t.shape[0]
        assert p.shape[1] == 3


# ── Velocity / acceleration bounds ──────────────────────────────────


class TestBoundsRespected:
    @pytest.mark.parametrize("ndof", [2, 3, 6])
    def test_velocity_within_limits(self, ndof):
        rng = np.random.default_rng(ndof)
        vel_limit = rng.uniform(0.5, 2.0, ndof)
        acc_limit = rng.uniform(1.0, 5.0, ndof)
        path = rng.normal(size=(8, ndof)) * 0.5
        param = TimeOptimalParameterizer(vel_limit, acc_limit)
        traj = param.parameterize(path)
        _, _, v, _ = traj.sample_uniform(dt=0.01)
        # Allow small numerical overshoot.
        assert np.all(np.abs(v) <= vel_limit * 1.01 + 1e-6)

    @pytest.mark.parametrize("ndof", [2, 3, 6])
    def test_acceleration_within_limits(self, ndof):
        rng = np.random.default_rng(100 + ndof)
        vel_limit = rng.uniform(0.5, 2.0, ndof)
        acc_limit = rng.uniform(1.0, 5.0, ndof)
        path = rng.normal(size=(8, ndof)) * 0.5
        param = TimeOptimalParameterizer(vel_limit, acc_limit)
        traj = param.parameterize(path)
        _, _, _, a = traj.sample_uniform(dt=0.01)
        # TOPP-RA discretizes a smooth path and may have tiny sampling
        # overshoot near grid transitions.
        assert np.all(np.abs(a) <= acc_limit * 1.02 + 1e-6)

    def test_scaling_slows_trajectory(self):
        param = TimeOptimalParameterizer(np.ones(3), np.ones(3) * 2)
        path = _zigzag_path(3)
        traj_fast = param.parameterize(path)
        traj_slow = param.parameterize(path, velocity_scaling=0.5)
        assert traj_slow.duration > traj_fast.duration


# ── One-shot convenience function ────────────────────────────────────


class TestParameterizePath:
    def test_one_shot(self):
        path = _zigzag_path(3)
        traj = parameterize_path(path, np.ones(3), np.ones(3) * 2)
        assert isinstance(traj, Trajectory)
        assert traj.duration > 0

    def test_one_shot_totg_method(self):
        pytest.importorskip("autolife_planning._time_parameterization")
        path = _zigzag_path(3)
        traj = parameterize_path(
            path,
            np.ones(3),
            np.ones(3) * 2,
            method="totg",
        )
        assert isinstance(traj, Trajectory)
        assert traj.duration > 0
