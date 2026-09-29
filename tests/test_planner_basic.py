"""End-to-end checks for the OMPL+VAMP planner without obstacles.

Mirrors the spirit of ``examples/subgroup_planning_example.py`` and
``examples/motion_planning_example.py`` (without the table) — small,
fast, and only exercises the bits the demos demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")


def test_available_robots_includes_known_subgroups():
    from autolife_planning.planning import available_robots

    names = available_robots()
    assert "autolife" in names
    assert "autolife_left_arm" in names
    assert "autolife_right_arm" in names
    assert "autolife_dual_arm" in names


def test_planner_dimension_matches_subgroup(left_arm_planner):
    # Left arm is the 7-DOF chain in PLANNING_SUBGROUPS.
    assert left_arm_planner._ndof == 7
    lo = np.asarray(left_arm_planner._planner.lower_bounds())
    hi = np.asarray(left_arm_planner._planner.upper_bounds())
    assert lo.shape == (7,) and hi.shape == (7,)
    assert np.all(hi > lo)


def test_extract_embed_round_trip(left_arm_planner, home_joints):
    extracted = left_arm_planner.extract_config(home_joints)
    embedded = left_arm_planner.embed_config(extracted)
    np.testing.assert_allclose(embedded, home_joints)


def test_home_is_collision_free(left_arm_planner, left_arm_start):
    assert left_arm_planner.validate(left_arm_start)


def test_sample_valid_returns_collision_free(left_arm_planner):
    cfg = left_arm_planner.sample_valid()
    assert cfg.shape == (7,)
    assert left_arm_planner.validate(cfg)


def test_trivial_plan_succeeds(left_arm_planner, left_arm_start):
    """Planning from a state to itself must always succeed."""
    result = left_arm_planner.plan(left_arm_start, left_arm_start)
    assert result.success
    assert result.path is not None and result.path.shape[1] == 7


def test_plan_to_random_valid_goal(left_arm_planner, left_arm_start):
    np.random.seed(0)
    goal = left_arm_planner.sample_valid()
    result = left_arm_planner.plan(left_arm_start, goal)
    # rrtc with 2 s + a free workspace should solve almost surely; if
    # the random sample lands in a tricky pocket we still want a clean
    # status, not a crash.
    assert result.status.value in {"success", "failed"}
    if result.success:
        assert result.path is not None
        np.testing.assert_allclose(result.path[0], left_arm_start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)


def test_plan_rejects_wrong_dimension(left_arm_planner, left_arm_start):
    with pytest.raises(ValueError):
        left_arm_planner.plan(left_arm_start, np.zeros(8))


def test_simplify_and_interpolate_path(home_joints):
    """Post-hoc simplify + interpolate round-trip on a real plan."""
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    # Plan without simplification/interpolation so the standalone
    # helpers have room to actually do something.
    raw_planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(
            planner_name="rrtc",
            time_limit=2.0,
            simplify=False,
            interpolate=False,
        ),
    )
    start = raw_planner.extract_config(home_joints)
    np.random.seed(0)
    goal = raw_planner.sample_valid()
    result = raw_planner.plan(start, goal)
    if not result.success:
        pytest.skip("rrtc didn't find a path for this random goal")

    n_raw = result.path.shape[0]

    # Simplify: endpoint-preserving, size can only shrink or stay.
    simp = raw_planner.simplify_path(result.path, time_limit=1.0)
    assert simp.shape[1] == 7
    assert simp.shape[0] <= n_raw
    np.testing.assert_allclose(simp[0], start, atol=1e-6)
    np.testing.assert_allclose(simp[-1], goal, atol=1e-6)

    # Interpolate by count — exact output size.
    dense_count = raw_planner.interpolate_path(simp, count=50, resolution=0.0)
    assert dense_count.shape == (50, 7)
    np.testing.assert_allclose(dense_count[0], simp[0], atol=1e-6)
    np.testing.assert_allclose(dense_count[-1], simp[-1], atol=1e-6)

    # Interpolate by resolution — denser than the simplified path.
    dense_res = raw_planner.interpolate_path(simp, count=0, resolution=64.0)
    assert dense_res.shape[1] == 7
    assert dense_res.shape[0] >= simp.shape[0]

    # Mutual-exclusion guardrail.
    with pytest.raises(ValueError):
        raw_planner.interpolate_path(simp, count=10, resolution=64.0)


def test_simplify_path_rejects_wrong_dimension(left_arm_planner):
    with pytest.raises(ValueError):
        left_arm_planner.simplify_path(np.zeros((4, 8)))


def test_interpolate_path_rejects_wrong_dimension(left_arm_planner):
    with pytest.raises(ValueError):
        left_arm_planner.interpolate_path(np.zeros((4, 8)))


def test_validate_batch_matches_single(left_arm_planner, left_arm_start):
    """Batched SIMD check must agree with per-config calls on every sample."""
    np.random.seed(0)
    lo = np.asarray(left_arm_planner._planner.lower_bounds())
    hi = np.asarray(left_arm_planner._planner.upper_bounds())
    # Spans across and past kRake=8: tail blocks, pure blocks, mixed validity.
    samples = np.random.uniform(lo, hi, size=(37, 7))
    # Anchor one known-valid row so the common "all-valid block" path runs.
    samples[0] = left_arm_start

    expected = np.array([left_arm_planner.validate(s) for s in samples], dtype=bool)
    got = left_arm_planner.validate_batch(samples)
    assert got.shape == (37,) and got.dtype == bool
    np.testing.assert_array_equal(got, expected)


def test_validate_batch_empty(left_arm_planner):
    out = left_arm_planner.validate_batch(np.zeros((0, 7)))
    assert out.shape == (0,) and out.dtype == bool


def test_validate_batch_rejects_wrong_dimension(left_arm_planner):
    with pytest.raises(ValueError):
        left_arm_planner.validate_batch(np.zeros((4, 8)))


def test_validate_batch_full_body_roundtrip(home_joints):
    """Full-body (24-DOF) batched check — no subgroup expansion path."""
    from autolife_planning._ompl_vamp import OmplVampPlanner

    planner = OmplVampPlanner()
    home = home_joints.tolist()
    out_of_bounds = [10.0] * 24

    batch = [home, out_of_bounds, home, out_of_bounds] * 3  # 12 = 1 full + 1 tail
    got = planner.validate_batch(batch)
    expected = [planner.validate(c) for c in batch]
    assert got == expected


def test_leg_torso_dual_arm_configs_collision_free(home_joints):
    """Pins down three configs for ``autolife_leg_torso_dual_arm`` that
    must validate as collision-free with the default HOME_JOINTS base.

    Added after a stale-build episode where a pre-rebuilt extension
    reported one of these as in-collision; this test guards the
    reproducibility of the fix (and guards against future sphere-geometry
    regressions on this subgroup).
    """
    from autolife_planning.planning import create_planner

    start = np.array(
        [
            -0.137867,
            -0.49979,
            -0.499886,
            0.397685,
            0.000191745,
            -0.000191745,
            0.000191745,
            0.0162983,
            0.000190738,
            -0.000190738,
            0.000572213,
            2.20142,
            1.90038,
            2.87982,
            -1.96864,
            2.81395,
            -0.359541,
            -0.0551232,
        ]
    )
    goal = np.array(
        [
            -0.137867,
            -0.499886,
            -0.499886,
            0.397685,
            0.349167,
            -0.000191745,
            0.000191745,
            1.91994,
            0.000190738,
            -0.000572213,
            -0.000190738,
            2.20142,
            1.90038,
            2.87982,
            -1.96864,
            2.81395,
            -0.359541,
            -0.0551232,
        ]
    )
    zero = np.zeros(18)

    planner = create_planner("autolife_leg_torso_dual_arm", base_config=home_joints)
    assert planner._ndof == 18
    assert planner.validate(start), "reported start config should be collision-free"
    assert planner.validate(goal), "reported goal config should be collision-free"
    assert planner.validate(zero), "all-zero 18-DOF config should be collision-free"
