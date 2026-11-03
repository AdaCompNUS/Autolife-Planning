"""Checks of the low-level ``OmplVampPlanner``.

Most use the base subgroup (x, y, theta), the rest of the body frozen
at home, among sphere obstacles.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")

START = np.array([-1.5, 0.0, 0.0])
GOAL = np.array([1.5, 0.0, 0.0])


def base_planner():
    """Base (x, y, theta) planner, the rest of the body frozen at home."""
    from autolife_planning._ompl_vamp import OmplVampPlanner
    from autolife_planning.autolife import HOME_JOINTS

    planner = OmplVampPlanner([0, 1, 2], HOME_JOINTS.tolist())
    lower, upper = OmplVampPlanner.default_joint_limits()
    lower[:2], upper[:2] = -3.0, 3.0
    planner.set_joint_limits(lower, upper)
    return planner


def test_base_subgroup_checks_the_frozen_body():
    # A small sphere at head height: only the frozen upper body reaches it.
    planner = base_planner()
    planner.add_sphere([2.0, 0.0, 1.59], 0.05)
    assert not planner.validate(np.array([2.0, 0.0, 0.0]))
    assert planner.validate(np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("name", ["pdst", "kpiece", "bkpiece", "lbkpiece", "stride"])
def test_partial_motions_stop_at_obstacles(name):
    """These planners grow from the last valid state of a blocked motion."""
    planner = base_planner()
    # A wall across the straight line from START to GOAL, |y| <= 1.6.
    for y in np.arange(-1.5, 1.51, 0.1):
        planner.add_sphere([0.0, y, 0.3], 0.1)
    for _ in range(5):
        result = planner.plan(START, GOAL, name, 5.0, False, False, 0.0)
        assert result.solved
        # The base centre crosses x = 0 only beyond the ends of the wall.
        a, b = result.path[:-1], result.path[1:]
        cross = np.flatnonzero(np.sign(a[:, 0]) != np.sign(b[:, 0]))
        y = a[cross, 1] + (b[cross, 1] - a[cross, 1]) * a[cross, 0] / (
            a[cross, 0] - b[cross, 0]
        )
        assert np.all(np.abs(y) > 1.5)


def test_unreachable_goal_is_not_solved():
    """RRT-Connect's closest approach to an enclosed goal is not a solution."""
    planner = base_planner()
    # A ring of spheres at wheel height around the goal.
    for a in np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False):
        planner.add_sphere([1.5 + 1.1 * np.cos(a), 1.1 * np.sin(a), 0.3], 0.1)
    assert planner.validate(START) and planner.validate(GOAL)
    result = planner.plan(START, GOAL, "rrtc", 0.5, True, True, 64.0)
    assert not result.solved
    assert result.path.shape == (0, 3)


def test_subgroup_inputs_are_checked(home_joints):
    from autolife_planning._ompl_vamp import OmplVampPlanner

    with pytest.raises(ValueError):
        OmplVampPlanner([0, 1, 2], home_joints[:20].tolist())
    with pytest.raises(ValueError):
        OmplVampPlanner([0, 1, 24], home_joints.tolist())
    planner = OmplVampPlanner()
    with pytest.raises(ValueError):
        planner.set_subgroup([0, 1, 2], home_joints.tolist() + [0.0])


def lattice(lower, upper, spacing):
    """Lattice points from ``lower`` to ``upper`` (m) and the grid's origin."""
    axes = [np.arange(lo, hi + spacing / 2, spacing) for lo, hi in zip(lower, upper)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1), [a[0] for a in axes]


def test_distance_grid_agrees_with_sphere_obstacles():
    """Away from the surface, a grid of the spheres' distance accepts and rejects
    what the spheres do.

    The spheres lie far apart, so trilinear interpolation of their distance only
    overestimates it, by at most 3 h^2 / (8 d) in a cell at distance d or more
    from a centre: under 1.5 mm for the robot's smallest sphere (1 cm) here.
    Configurations clear of the spheres by ``tol`` are free in the grid; those
    that overlap them by more than ``tol`` collide.
    """
    centers = np.array([[0.8, 0.0, 1.0], [-0.6, 0.7, 0.4], [0.0, -0.9, 1.5]])
    radius, spacing, tol = 0.15, 0.02, 2e-3
    points, origin = lattice([-3.0, -3.0, -0.5], [3.0, 3.0, 2.5], spacing)
    distances = np.linalg.norm(points[..., None, :] - centers, axis=-1).min(-1) - radius
    grid = base_planner()
    grid.set_distance_grid(distances.astype(np.float32), origin, [spacing] * 3)
    inflated, deflated = base_planner(), base_planner()
    for center in centers:
        inflated.add_sphere(center.tolist(), radius + tol)
        deflated.add_sphere(center.tolist(), radius - tol)

    rng = np.random.default_rng(0)
    q = rng.uniform([-1.0, -1.0, -np.pi], [1.0, 1.0, np.pi], (4000, 3))
    valid, free, colliding = (
        grid.validate_batch(q),
        inflated.validate_batch(q),
        ~deflated.validate_batch(q),
    )
    assert valid[free].all()
    assert not valid[colliding].any()
    assert 0.2 < free.mean() < 0.8


def test_chassis_reads_the_floor_free_field(home_joints):
    """The chassis rests on the floor; every other link collides with it."""
    from autolife_planning._ompl_vamp import OmplVampPlanner

    spacing = 0.05
    points, origin = lattice([-2.0, -2.0, -0.5], [2.0, 2.0, 2.5], spacing)
    no_floor = np.full(points.shape[:3], 10.0, dtype=np.float32)
    planner = OmplVampPlanner()
    home = home_joints.copy()

    def floor_at(height, without_floor):
        planner.set_distance_grid(
            (points[..., 2] - height).astype(np.float32),
            origin,
            [spacing] * 3,
            without_floor,
        )
        return planner.validate(home)

    assert not floor_at(0.0, None)
    assert floor_at(0.0, no_floor)
    # A floor at 1 m: the legs and waist read it, the chassis does not.
    assert not floor_at(1.0, no_floor)
    planner.set_distance_grid(no_floor, origin, [spacing] * 3)
    assert planner.validate(home)
    # A lattice from z = 0: the chassis reaches below it, which only the
    # floor-free field's box allows.
    points, origin = lattice([-2.0, -2.0, 0.0], [2.0, 2.0, 2.5], spacing)
    no_floor = np.full(points.shape[:3], 10.0, dtype=np.float32)
    planner.set_distance_grid(no_floor, origin, [spacing] * 3)
    assert not planner.validate(home)
    planner.set_distance_grid(no_floor, origin, [spacing] * 3, no_floor)
    assert planner.validate(home)


def test_distance_grid_box_is_the_workspace():
    """A base whose spheres leave the grid's box collides."""
    spacing = 0.05
    points, origin = lattice([-1.0, -1.0, -0.5], [1.0, 1.0, 2.5], spacing)
    planner = base_planner()
    planner.set_distance_grid(
        np.full(points.shape[:3], 10.0, dtype=np.float32), origin, [spacing] * 3
    )
    assert planner.validate(np.array([0.0, 0.0, 0.0]))
    assert not planner.validate(np.array([1.5, 0.0, 0.0]))


def test_distance_grid_inputs_are_checked():
    planner = base_planner()
    field = np.zeros((4, 5, 6), dtype=np.float32)
    origin, spacing = [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]
    assert not planner.has_distance_grid()
    with pytest.raises(ValueError):
        planner.set_distance_grid(field, origin, spacing, field[:, :, :5].copy())
    with pytest.raises(ValueError):
        planner.set_distance_grid(field, origin, [0.1, 0.0, 0.1])
    with pytest.raises(ValueError):
        planner.set_distance_grid(field[:1].copy(), origin, spacing)
    assert not planner.has_distance_grid()
    planner.set_distance_grid(field, origin, spacing, field)
    assert planner.has_distance_grid()
    assert planner.remove_distance_grid()
    assert not planner.remove_distance_grid()
    planner.set_distance_grid(field, origin, spacing)
    planner.clear_environment()
    assert not planner.has_distance_grid()


def test_distance_grid_reads_the_arrays_in_place():
    """The planner reads the caller's array without a copy and keeps it alive."""
    spacing = 0.05
    points, origin = lattice([-1.0, -1.0, -0.5], [1.0, 1.0, 2.5], spacing)
    field = np.full(points.shape[:3], 10.0, dtype=np.float32)
    planner = base_planner()
    planner.set_distance_grid(field, origin, [spacing] * 3)
    assert planner.validate(np.array([0.0, 0.0, 0.0]))
    field[...] = -1.0
    assert not planner.validate(np.array([0.0, 0.0, 0.0]))
    del field
    assert not planner.validate(np.array([0.0, 0.0, 0.0]))
