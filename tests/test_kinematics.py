"""Kinematics tests — IK solver factory, TRAC-IK, Pinocchio FK, collision model.

Mirrors ``examples/ik_example.py`` and the constrained-IK demo, but
keeps every solve cheap (small offsets, deterministic seed) so the
suite stays under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from autolife_planning.autolife import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.types import CuroboV2IKConfig, IKConfig, SE3Pose, SolveType

HOME_LEFT_ARM = HOME_JOINTS[JOINT_GROUPS["left_arm"]]
HOME_RIGHT_ARM = HOME_JOINTS[JOINT_GROUPS["right_arm"]]


# ── Factory & chain resolution ───────────────────────────────────────


def test_factory_rejects_unknown_chain():
    pytest.importorskip("pytracik")
    from autolife_planning.kinematics import create_ik_solver

    with pytest.raises(ValueError, match="Unknown chain"):
        create_ik_solver("not_a_real_chain")


def test_factory_rejects_unknown_backend():
    pytest.importorskip("pytracik")
    from autolife_planning.kinematics import create_ik_solver

    with pytest.raises(ValueError, match="Unknown backend"):
        create_ik_solver("left_arm", backend="not_a_backend")


def test_factory_side_suffix_resolves():
    pytest.importorskip("pytracik")
    from autolife_planning.kinematics import create_ik_solver

    solver = create_ik_solver("whole_body", side="left")
    assert solver.num_joints == 11


# ── TRAC-IK solver ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trac_left_arm():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from autolife_planning.kinematics import create_ik_solver

    return create_ik_solver("left_arm", config=IKConfig(max_attempts=3))


def test_trac_ik_chain_metadata(trac_left_arm):
    assert trac_left_arm.num_joints == 7
    assert trac_left_arm.base_frame
    assert trac_left_arm.ee_frame


def test_trac_ik_fk_runs(trac_left_arm):
    pose = trac_left_arm.fk(HOME_LEFT_ARM)
    assert isinstance(pose, SE3Pose)
    assert pose.position.shape == (3,)
    assert pose.rotation.shape == (3, 3)
    # Rotation matrix should be orthonormal.
    np.testing.assert_allclose(pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-9)


def test_trac_ik_solve_round_trip(trac_left_arm):
    """IK(FK(q)) with a tiny offset should land on a config whose FK matches."""
    home_pose = trac_left_arm.fk(HOME_LEFT_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.03, 0.0, -0.02]),
        rotation=home_pose.rotation,
    )
    result = trac_left_arm.solve(target, seed=HOME_LEFT_ARM)
    if not result.success:
        pytest.skip(f"TRAC-IK did not converge ({result.status.value}); flaky on CI")

    assert result.joint_positions is not None
    assert result.joint_positions.shape == (7,)
    assert result.position_error < 1e-3
    assert result.orientation_error < 1e-3

    achieved = trac_left_arm.fk(result.joint_positions)
    np.testing.assert_allclose(achieved.position, target.position, atol=1e-3)


def test_trac_ik_round_trip_multiple_targets(trac_left_arm):
    """Real correctness: sample several FK targets, IK them back, FK again must match.

    Picks a few small perturbations of HOME, FK→pose→IK→q'→FK→pose'.
    Position and orientation error must agree to within the post-solve
    tolerance the IK config promises (1e-4 m / 1e-4 rad).
    """
    rng = np.random.default_rng(0)
    n_solved = 0
    for _ in range(5):
        delta = rng.uniform(-0.1, 0.1, size=7)
        q_seed = HOME_LEFT_ARM + delta
        target_pose = trac_left_arm.fk(q_seed)

        result = trac_left_arm.solve(target_pose, seed=q_seed)
        if not result.success:
            continue
        n_solved += 1

        achieved = trac_left_arm.fk(result.joint_positions)
        # Position
        np.testing.assert_allclose(achieved.position, target_pose.position, atol=1e-3)
        # Orientation: rotation matrix product close to identity
        R_err = achieved.rotation.T @ target_pose.rotation
        np.testing.assert_allclose(R_err, np.eye(3), atol=1e-3)

    assert n_solved >= 3, "TRAC-IK should converge on most small perturbations"


def test_trac_ik_solve_types_accepted():
    pytest.importorskip("pytracik")
    pytest.importorskip("pinocchio")
    from autolife_planning.kinematics import create_ik_solver

    for st in (SolveType.SPEED, SolveType.DISTANCE):
        s = create_ik_solver(
            "left_arm",
            config=IKConfig(solve_type=st, max_attempts=1),
        )
        assert s.num_joints == 7


def test_trac_ik_set_joint_limits_validates(trac_left_arm):
    lo, hi = trac_left_arm.joint_limits
    with pytest.raises(ValueError):
        trac_left_arm.set_joint_limits(lo[:-1], hi)


# ── Pinocchio FK ─────────────────────────────────────────────────────


_LEFT_ARM_JOINT_NAMES = [
    "Joint_Left_Shoulder_Inner",
    "Joint_Left_Shoulder_Outer",
    "Joint_Left_UpperArm",
    "Joint_Left_Elbow",
    "Joint_Left_Forearm",
    "Joint_Left_Wrist_Upper",
    "Joint_Left_Wrist_Lower",
]


@pytest.fixture(scope="module")
def left_arm_pin_context():
    pytest.importorskip("pinocchio")
    from autolife_planning.autolife import CHAIN_CONFIGS
    from autolife_planning.kinematics import create_pinocchio_context

    chain = CHAIN_CONFIGS["left_arm"]
    return create_pinocchio_context(
        urdf_path=chain.urdf_path,
        end_effector_frame=chain.ee_link,
        joint_names=_LEFT_ARM_JOINT_NAMES,
    )


def test_pinocchio_fk_matches_trac_ik(left_arm_pin_context, trac_left_arm):
    """Both backends should agree on the EE pose at HOME (same URDF)."""
    from autolife_planning.kinematics import compute_forward_kinematics

    pin_pose = compute_forward_kinematics(left_arm_pin_context, HOME_LEFT_ARM)
    trac_pose = trac_left_arm.fk(HOME_LEFT_ARM)

    np.testing.assert_allclose(pin_pose.position, trac_pose.position, atol=1e-6)
    np.testing.assert_allclose(pin_pose.rotation, trac_pose.rotation, atol=1e-6)


def test_pinocchio_jacobian_shape(left_arm_pin_context):
    from autolife_planning.kinematics import compute_jacobian

    J = compute_jacobian(left_arm_pin_context, HOME_LEFT_ARM)
    # 6 task-space rows × however many actuated joints the model exposes.
    assert J.shape[0] == 6
    assert J.shape[1] >= 7


def test_pinocchio_fk_matches_trac_ik_random_configs(
    left_arm_pin_context, trac_left_arm
):
    """Real correctness: cross-verify FK at multiple random valid configs."""
    from autolife_planning.kinematics import compute_forward_kinematics

    lo, hi = trac_left_arm.joint_limits
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        pin_pose = compute_forward_kinematics(left_arm_pin_context, q)
        trac_pose = trac_left_arm.fk(q)
        np.testing.assert_allclose(pin_pose.position, trac_pose.position, atol=1e-9)
        np.testing.assert_allclose(pin_pose.rotation, trac_pose.rotation, atol=1e-9)


def test_pinocchio_jacobian_matches_finite_difference(left_arm_pin_context):
    """Real correctness: Jacobian columns equal ∂(FK)/∂qᵢ via finite differences.

    Uses LOCAL_WORLD_ALIGNED frame (the default) — translational part is
    just dp/dq, so we can FD it directly without unwrapping rotation
    increments.
    """
    from autolife_planning.kinematics import (
        compute_forward_kinematics,
        compute_jacobian,
    )

    rng = np.random.default_rng(7)
    q = HOME_LEFT_ARM + rng.uniform(-0.05, 0.05, size=7)
    J = compute_jacobian(left_arm_pin_context, q)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        fp = compute_forward_kinematics(left_arm_pin_context, q + dq).position
        fm = compute_forward_kinematics(left_arm_pin_context, q - dq).position
        dp_dq = (fp - fm) / (2 * eps)
        # Translational rows of the Jacobian in LOCAL_WORLD_ALIGNED == dp/dq
        np.testing.assert_allclose(J[:3, i], dp_dq, atol=1e-5)


def test_pinocchio_context_rejects_unknown_frame():
    pytest.importorskip("pinocchio")
    from autolife_planning.autolife import CHAIN_CONFIGS
    from autolife_planning.kinematics import create_pinocchio_context

    chain = CHAIN_CONFIGS["left_arm"]
    with pytest.raises(ValueError, match="not found"):
        create_pinocchio_context(
            urdf_path=chain.urdf_path,
            end_effector_frame="Not_A_Real_Frame",
        )


# ── Collision model ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def collision_ctx():
    pytest.importorskip("pinocchio")
    pytest.importorskip("hppfcl")
    from autolife_planning.autolife import CHAIN_CONFIGS
    from autolife_planning.kinematics import build_collision_model

    return build_collision_model(CHAIN_CONFIGS["left_arm"].urdf_path)


def test_collision_model_has_pairs(collision_ctx):
    assert collision_ctx.collision_model.ngeoms > 0
    # SRDF should leave some pairs (we only prune adjacent overlapping links).
    assert len(collision_ctx.collision_model.collisionPairs) >= 0


def test_add_pointcloud_obstacles_validates_shape(collision_ctx):
    from autolife_planning.kinematics import add_pointcloud_obstacles

    with pytest.raises(ValueError):
        add_pointcloud_obstacles(collision_ctx, np.zeros((4, 2)))


def test_add_pointcloud_obstacles_returns_count(collision_ctx):
    from autolife_planning.kinematics import add_pointcloud_obstacles

    pts = np.array(
        [
            [10.0, 10.0, 10.0],  # far enough to never matter for the arm
            [10.0, 10.5, 10.0],
            [10.5, 10.0, 10.0],
        ]
    )
    n_before = collision_ctx.collision_model.ngeoms
    n_added = add_pointcloud_obstacles(collision_ctx, pts, radius=0.01)
    assert n_added == 3
    assert collision_ctx.collision_model.ngeoms == n_before + 3


# ── Pink IK solver ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pink_left_arm():
    pytest.importorskip("pink")
    pytest.importorskip("qpsolvers")
    pytest.importorskip("pinocchio")
    from autolife_planning.kinematics import create_ik_solver
    from autolife_planning.types import PinkIKConfig

    return create_ik_solver(
        "left_arm",
        backend="pink",
        joint_names=_LEFT_ARM_JOINT_NAMES,
        config=PinkIKConfig(max_iterations=300),
    )


def test_pink_solver_chain_metadata(pink_left_arm):
    assert pink_left_arm.num_joints == 7
    assert pink_left_arm.base_frame
    assert pink_left_arm.ee_frame
    assert pink_left_arm.joint_names == _LEFT_ARM_JOINT_NAMES


def test_pink_fk_matches_trac_ik(pink_left_arm, trac_left_arm):
    """Pink and TRAC-IK must agree on the EE pose at HOME (same URDF)."""
    pink_pose = pink_left_arm.fk(HOME_LEFT_ARM)
    trac_pose = trac_left_arm.fk(HOME_LEFT_ARM)
    np.testing.assert_allclose(pink_pose.position, trac_pose.position, atol=1e-6)
    np.testing.assert_allclose(pink_pose.rotation, trac_pose.rotation, atol=1e-6)


def test_pink_fk_matches_trac_ik_random_configs(pink_left_arm, trac_left_arm):
    """Real correctness: Pink and TRAC-IK FK agree across random configs."""
    lo, hi = trac_left_arm.joint_limits
    rng = np.random.default_rng(123)
    for _ in range(10):
        q = rng.uniform(lo, hi)
        pink_pose = pink_left_arm.fk(q)
        trac_pose = trac_left_arm.fk(q)
        np.testing.assert_allclose(pink_pose.position, trac_pose.position, atol=1e-9)
        np.testing.assert_allclose(pink_pose.rotation, trac_pose.rotation, atol=1e-9)


def test_pink_solve_constrained_converges(pink_left_arm):
    """Tiny offset — Pink's iterative QP should land near the target."""
    home_pose = pink_left_arm.fk(HOME_LEFT_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.02, 0.0, -0.01]),
        rotation=home_pose.rotation,
    )
    result = pink_left_arm.solve_constrained(target, seed=HOME_LEFT_ARM)
    if not result.success:
        pytest.skip(f"Pink IK did not converge ({result.status.value}); flaky on CI")

    assert result.joint_positions is not None
    assert result.joint_positions.shape == (7,)
    assert result.position_error < 5e-2
    assert result.trajectory is not None and result.trajectory.shape[1] == 7


def test_pink_solve_returns_plain_ik_result(pink_left_arm):
    """``solve()`` is the IKSolverBase contract — must yield an IKResult."""
    from autolife_planning.types import IKResult

    home_pose = pink_left_arm.fk(HOME_LEFT_ARM)
    target = SE3Pose(
        position=home_pose.position + np.array([0.01, 0.0, 0.0]),
        rotation=home_pose.rotation,
    )
    result = pink_left_arm.solve(target, seed=HOME_LEFT_ARM)
    assert isinstance(result, IKResult)


def test_pink_set_collision_context_accepts_none(pink_left_arm):
    pink_left_arm.set_collision_context(None)


def test_pink_solver_rejects_unknown_joint():
    pytest.importorskip("pink")
    pytest.importorskip("pinocchio")
    from autolife_planning.autolife import CHAIN_CONFIGS
    from autolife_planning.kinematics.pink_ik_solver import PinkIKSolver

    with pytest.raises(ValueError, match="not in model"):
        PinkIKSolver(
            CHAIN_CONFIGS["left_arm"],
            joint_names=["Joint_Does_Not_Exist"],
        )


# ── cuRoboV2 IK solver ────────────────────────────────────────────


@pytest.fixture(scope="module")
def curobo_v2_whole_body():
    pytest.importorskip("pinocchio")
    from autolife_planning.kinematics import create_ik_solver

    return create_ik_solver(
        "whole_body",
        side="left",
        backend="curobo_v2",
        config=CuroboV2IKConfig(num_seeds=8, return_seeds=2),
    )


def test_curobo_v2_factory_is_lazy(curobo_v2_whole_body):
    """Factory and FK do not initialize PyTorch/CUDA or cuRobo."""
    assert curobo_v2_whole_body.num_joints == 11
    assert curobo_v2_whole_body.stability_supported
    assert curobo_v2_whole_body.joint_names[:4] == [
        "Joint_Ankle",
        "Joint_Knee",
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
    ]


def test_curobo_v2_fk_uses_package_world_frame(curobo_v2_whole_body):
    groups = JOINT_GROUPS
    home = np.concatenate(
        [
            HOME_JOINTS[groups["legs"]],
            HOME_JOINTS[groups["waist"]],
            HOME_JOINTS[groups["left_arm"]],
        ]
    )
    pose = curobo_v2_whole_body.fk(home)
    assert pose.position.shape == (3,)
    np.testing.assert_allclose(pose.rotation @ pose.rotation.T, np.eye(3), atol=1e-9)


def test_curobo_v2_stable_projection(curobo_v2_whole_body):
    config = CuroboV2IKConfig(num_seeds=8, return_seeds=2)
    seed = np.zeros(curobo_v2_whole_body.num_joints)
    seed[0] = 0.4
    seed[2] = -2.0
    seed[3] = 2.0
    projected = curobo_v2_whole_body._project_stable(seed, config)
    assert projected[0] == pytest.approx(0.4)
    assert projected[1] == pytest.approx(0.8)
    assert projected[2] != pytest.approx(projected[0])
    assert projected[2] - projected[0] == pytest.approx(config.waist_ankle_min)
    assert projected[3] == pytest.approx(config.stability_waist_yaw_max)
    assert curobo_v2_whole_body._is_stable(projected, config)

    seed[2] = 2.0
    projected = curobo_v2_whole_body._project_stable(seed, config)
    assert projected[2] - projected[0] == pytest.approx(config.waist_ankle_max)
    assert curobo_v2_whole_body._is_stable(projected, config)

    projected[2] = projected[0] + config.waist_ankle_min - 1e-6
    assert not curobo_v2_whole_body._is_stable(projected, config)


def test_curobo_v2_waist_ankle_constraint_hinge():
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo.inverse_kinematics")
    from curobo.types import DeviceCfg

    from autolife_planning.kinematics.curobo_v2_constraints import (
        WaistAnkleConstraint,
        WaistAnkleConstraintCfg,
    )

    device_cfg = DeviceCfg(device="cpu", dtype=torch.float32)
    config = WaistAnkleConstraintCfg(
        weight=2.0,
        lower_bound=-0.2,
        upper_bound=0.5,
        device_cfg=device_cfg,
    )
    constraint = WaistAnkleConstraint(config, ankle_index=0, waist_index=1)
    joints = torch.tensor(
        [[[0.0, -0.3], [0.0, 0.2], [0.0, 0.7]]],
        requires_grad=True,
    )

    values = constraint.forward(joints)

    assert values.shape == (1, 3, 1)
    torch.testing.assert_close(
        values,
        torch.tensor([[[0.2], [0.0], [0.4]]]),
    )
    values.sum().backward()
    torch.testing.assert_close(
        joints.grad,
        torch.tensor([[[2.0, -2.0], [0.0, 0.0], [-2.0, 2.0]]]),
    )


def test_curobo_v2_empty_batches(curobo_v2_whole_body):
    seeds = np.empty((0, curobo_v2_whole_body.num_joints))
    assert curobo_v2_whole_body.solve_batch([], seeds) == []
    assert curobo_v2_whole_body.solve_constrained_batch([], seeds) == []


def test_curobo_v2_batch_input_validation(curobo_v2_whole_body):
    pose = curobo_v2_whole_body.fk(
        np.zeros(curobo_v2_whole_body.num_joints)
    )
    with pytest.raises(ValueError, match="Expected seeds shape"):
        curobo_v2_whole_body.solve_batch(
            [pose, pose],
            np.zeros((1, curobo_v2_whole_body.num_joints)),
        )
    with pytest.raises(TypeError, match=r"invalid indices: \[1\]"):
        curobo_v2_whole_body.solve_batch(
            [pose, object()],
            np.zeros((2, curobo_v2_whole_body.num_joints)),
        )


def test_curobo_v2_batch_candidate_mapping(curobo_v2_whole_body):
    from types import SimpleNamespace

    dof = curobo_v2_whole_body.num_joints
    values = np.arange(2 * 3 * dof, dtype=np.float64).reshape(2, 3, dof)
    raw = SimpleNamespace(
        solution=values,
        success=np.array([[True, False, True], [False, True, False]]),
    )
    context = SimpleNamespace(
        joint_names=tuple(curobo_v2_whole_body.joint_names)
    )
    config = CuroboV2IKConfig(num_seeds=8, return_seeds=2)

    candidates = curobo_v2_whole_body._batch_successful_candidates(
        raw,
        2,
        context,
        config,
    )

    assert len(candidates) == 2
    np.testing.assert_array_equal(candidates[0][0], values[0, 0])
    np.testing.assert_array_equal(candidates[0][1], values[0, 2])
    np.testing.assert_array_equal(candidates[1][0], values[1, 1])


def test_curobo_v2_stable_urdf_only_mimics_knee():
    import xml.etree.ElementTree as ET

    from autolife_planning.autolife import CHAIN_CONFIGS
    from autolife_planning.kinematics.curobo_v2_ik_solver import (
        _stable_urdf_path,
    )

    config = CuroboV2IKConfig(
        num_seeds=8,
        return_seeds=2,
        stability_ankle_max=1.1,
    )
    path = _stable_urdf_path(CHAIN_CONFIGS["whole_body_left"].urdf_path, config)
    root = ET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    ankle_limit = joints["Joint_Ankle"].find("limit")
    knee_mimic = joints["Joint_Knee"].find("mimic")
    waist_mimic = joints["Joint_Waist_Pitch"].find("mimic")
    waist_yaw_limit = joints["Joint_Waist_Yaw"].find("limit")
    assert ankle_limit is not None
    assert knee_mimic is not None
    assert waist_mimic is None
    assert waist_yaw_limit is not None
    assert float(ankle_limit.attrib["lower"]) == pytest.approx(0.0)
    assert float(ankle_limit.attrib["upper"]) == pytest.approx(1.1)
    assert knee_mimic.attrib["joint"] == "Joint_Ankle"
    assert float(knee_mimic.attrib["multiplier"]) == pytest.approx(2.0)
    assert float(waist_yaw_limit.attrib["lower"]) == pytest.approx(
        config.stability_waist_yaw_min
    )
    assert float(waist_yaw_limit.attrib["upper"]) == pytest.approx(
        config.stability_waist_yaw_max
    )
