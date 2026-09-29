"""TOPP-RA time parameterization: analytic cases and limit checks.

Straight lines have closed-form time-optimal durations (trapezoidal or
triangular velocity profiles).  For general paths every trajectory is
densely resampled and checked against the joint limits, its endpoints,
and its distance from the piecewise-linear input path.
"""

import numpy as np
import pytest

from autolife_planning.autolife import HOME_JOINTS, MAX_ACCELERATION, MAX_VELOCITY
from autolife_planning.trajectory import TimeOptimalParameterizer, parameterize_path

DT = 1e-3
LIMIT_TOL = 1e-2  # relative, from TOPP-RA's discretization grid


def straight_line_duration(d, v, a):
    """Rest-to-rest time-optimal duration over distance d."""
    return d / v + v / a if d >= v * v / a else 2.0 * np.sqrt(d / a)


def polyline_distance(q, path):
    """Distance from each row of q to the nearest point of the polyline."""
    dist = np.full(len(q), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        t = np.clip(((q - a) @ ab) / (ab @ ab), 0.0, 1.0)
        dist = np.minimum(dist, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
    return dist


def assert_executable(traj, path, vel, acc):
    times, q, qd, qdd = traj.sample_uniform(DT)
    assert times[0] == 0.0 and times[-1] == pytest.approx(traj.duration)
    np.testing.assert_allclose(q[0], path[0], atol=1e-9)
    np.testing.assert_allclose(q[-1], path[-1], atol=1e-9)
    np.testing.assert_allclose(qd[[0, -1]], 0.0, atol=1e-9)
    assert np.all(np.abs(qd) <= vel * (1 + LIMIT_TOL))
    assert np.all(np.abs(qdd) <= acc * (1 + LIMIT_TOL))
    # Velocity is the derivative of position.
    np.testing.assert_allclose(
        np.diff(q, axis=0) / np.diff(times)[:, None],
        0.5 * (qd[1:] + qd[:-1]),
        atol=acc.max() * DT,
    )
    return q


@pytest.mark.parametrize("d", [0.5, 2.0])
def test_straight_line_matches_closed_form(d):
    traj = parameterize_path(np.array([[0.0], [d]]), [1.0], [0.5])
    assert traj.duration == pytest.approx(straight_line_duration(d, 1.0, 0.5))


def test_slowest_joint_sets_duration():
    # Joint 1 needs 2 / 0.5 + 0.5 / 1.0 = 4.5 s; joint 0 alone only 2.5 s.
    path = np.array([[0.0, 0.0], [2.0, 2.0]])
    traj = parameterize_path(path, [1.0, 0.5], [1.0, 1.0])
    assert traj.duration == pytest.approx(4.5, rel=1e-3)
    assert_executable(traj, path, np.array([1.0, 0.5]), np.array([1.0, 1.0]))


def test_scaling_slows_down():
    param = TimeOptimalParameterizer([1.0], [0.5])
    path = np.array([[0.0], [2.0]])
    assert param.parameterize(path, velocity_scaling=0.5).duration == pytest.approx(
        straight_line_duration(2.0, 0.5, 0.5)
    )
    assert param.parameterize(path, acceleration_scaling=0.5).duration == pytest.approx(
        straight_line_duration(2.0, 1.0, 0.25)
    )


def test_duplicate_waypoints():
    path = np.array([[0.0], [0.0], [2.0], [2.0]])
    assert parameterize_path(path, [1.0], [0.5]).duration == pytest.approx(4.0)


@pytest.mark.parametrize("knot_spacing", [0.05, 0.1, 0.2])
def test_random_paths_respect_limits_and_stay_close(knot_spacing):
    rng = np.random.default_rng(0)
    for _ in range(5):
        path = np.cumsum(rng.uniform(-0.5, 0.5, size=(6, 8)), axis=0)
        vel = rng.uniform(0.5, 1.5, 8)
        acc = rng.uniform(1.0, 3.0, 8)
        traj = parameterize_path(path, vel, acc, knot_spacing=knot_spacing)
        q = assert_executable(traj, path, vel, acc)
        assert polyline_distance(q, path).max() <= 0.2 * knot_spacing
        # The spline passes through every waypoint.
        _, q_fine, _, _ = traj.sample_uniform(1e-4)
        for w in path:
            assert np.linalg.norm(q_fine - w, axis=1).min() <= 1e-3


def test_sampling_agrees():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    traj = parameterize_path(path, [1.0, 1.0], [1.0, 1.0])
    times = np.linspace(0.0, traj.duration, 7)
    q, qd, qdd = traj.sample(times)
    for i, t in enumerate(times):
        np.testing.assert_allclose(traj.position(t), q[i])
        np.testing.assert_allclose(traj.velocity(t), qd[i])
        np.testing.assert_allclose(traj.acceleration(t), qdd[i])


def test_autolife_arm_plan():
    """Time a planned Autolife arm motion with the robot's joint limits."""
    pytest.importorskip("autolife_planning._ompl_vamp")
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(planner_name="rrtc", interpolate=False),
    )
    # An obstacle where the gripper would pass on the straight-line path,
    # so the plan has to detour around it.
    planner._planner.add_sphere([0.323, 0.25, 1.045], 0.05)
    start = planner.extract_config(HOME_JOINTS)
    goal = start + np.array([-0.9, 0.2, 0.3, -0.8, 0.3, 0.2, 0.0])
    assert not planner.validate_batch(np.linspace(start, goal, 50)).all()
    result = planner.plan(start, goal)
    assert result.success

    idx = planner.subgroup_indices
    vel = MAX_VELOCITY[idx]
    acc = MAX_ACCELERATION[idx]
    traj = TimeOptimalParameterizer(vel, acc).parameterize(result.path)
    q = assert_executable(traj, result.path, vel, acc)
    assert polyline_distance(q, result.path).max() <= 0.02
