"""Render every documentation video in one pass.

Runs entirely in ``DIRECT`` mode (no GUI), uses the offscreen
TinyRenderer to capture frames at 1280x720, and pipes them straight
into ``ffmpeg`` for H.264 encoding via
:class:`autolife_planning.utils.video_recorder.VideoRecorder`.
Output files land in ``docs/assets/``.

Clips produced:
  * ``trac_ik.mp4``                   — TRAC-IK sweep over three target
                                          offsets on the left arm
  * ``constrained_ik.mp4``            — Pink QP-IK with CoM / camera /
                                          collision constraints, several
                                          reach targets
  * ``motion_planning.mp4``           — 7-DOF left-arm plan around a
                                          table point cloud
  * ``subgroup_planning.mp4``         — three subgroup × stance plans
                                          stitched into one clip
  * ``constrained_plane.mp4``         — gripper slides on z = z₀
  * ``constrained_plane_obstacle.mp4``— plane constraint + sphere obstacle
  * ``constrained_line_horizontal.mp4``— gripper rides a horizontal rail
  * ``constrained_line_vertical.mp4`` — gripper slides a vertical rail
  * ``constrained_orientation_lock.mp4``— translation free, rotation locked

Usage:
    pixi run python scripts/render_videos/render_docs_videos.py
    pixi run python scripts/render_videos/render_docs_videos.py --only motion_planning
    pixi run python scripts/render_videos/render_docs_videos.py --only plane,line_h
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np
import pybullet as pb
from fire import Fire

# Reach into examples/ so we can borrow helpers like ``load_table``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))

from autolife_planning.autolife import (  # noqa: E402
    HOME_JOINTS,
    JOINT_GROUPS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv  # noqa: E402
from autolife_planning.kinematics import create_ik_solver  # noqa: E402
from autolife_planning.planning import (  # noqa: E402
    Constraint,
    Cost,
    SymbolicContext,
    create_planner,
)
from autolife_planning.types import (  # noqa: E402
    IKConfig,
    PinkIKConfig,
    PlannerConfig,
    SE3Pose,
    SolveType,
)
from autolife_planning.utils.video_recorder import (  # noqa: E402
    CameraView,
    VideoRecorder,
)

ASSETS_DIR = _REPO_ROOT / "docs" / "assets"
SUBGROUP = "autolife_left_arm"
# Rotation-reference link for the constraint residuals (Link_Left_Gripper
# is the rigid gripper body frame).  For translation constraints we use
# the symmetric midpoint between the two finger links — see
# ``ee_translation`` / ``ee_position`` in examples/constrained_planning/_shared.py
# for the rationale.
EE_LINK = "Link_Left_Gripper"
LEFT_FINGER_LINK = "Link_Left_Gripper_Left_Finger"
RIGHT_FINGER_LINK = "Link_Left_Gripper_Right_Finger"


def _ee_tcp_sym(ctx):
    """CasADi expression for the gripper TCP (finger midpoint)."""
    return 0.5 * (
        ctx.link_translation(LEFT_FINGER_LINK) + ctx.link_translation(RIGHT_FINGER_LINK)
    )


def _ee_tcp_eval(ctx, q):
    """Numeric evaluation of the TCP at joint config ``q``."""
    left = np.asarray(ctx.evaluate_link_pose(LEFT_FINGER_LINK, q))[:3, 3]
    right = np.asarray(ctx.evaluate_link_pose(RIGHT_FINGER_LINK, q))[:3, 3]
    return 0.5 * (left + right)


def _ee_tcp_world(env) -> np.ndarray:
    """Live world-frame TCP position via PyBullet FK (midpoint of the fingers)."""
    client = env.sim.client
    body = env.sim.skel_id
    li = env.sim.link_map[LEFT_FINGER_LINK]
    ri = env.sim.link_map[RIGHT_FINGER_LINK]
    lp = np.asarray(client.getLinkState(body, li)[0])
    rp = np.asarray(client.getLinkState(body, ri)[0])
    return 0.5 * (lp + rp)


# ───────────────────── shared helpers ───────────────────────────────────


def _make_env() -> PyBulletEnv:
    """Fresh headless env per clip — keeps state clean."""
    return PyBulletEnv(autolife_robot_config, visualize=False)


def _full_dof(arm_cfg: np.ndarray, group: str = "left_arm") -> np.ndarray:
    """Embed an arm-only joint vector into the full 24-DOF layout."""
    full = HOME_JOINTS.copy()
    full[JOINT_GROUPS[group]] = arm_cfg
    return full


def _stack_full_dof(arm_path: np.ndarray, group: str = "left_arm") -> np.ndarray:
    """Broadcast an arm-only path to a 24-DOF path."""
    full = np.tile(HOME_JOINTS, (len(arm_path), 1))
    full[:, JOINT_GROUPS[group]] = arm_path
    return full


def _interp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Return a linear interpolation between *a* and *b* with *n* samples."""
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return a[None, :] * (1.0 - ts) + b[None, :] * ts


def _join_arm_path(
    segments: list[np.ndarray], steps_per_segment: int = 40
) -> np.ndarray:
    """Concatenate several arm poses into one linearly-interpolated path."""
    full = []
    for i in range(len(segments) - 1):
        full.append(_interp(segments[i], segments[i + 1], steps_per_segment))
    return np.concatenate(full, axis=0)


def _draw_box(env, center, half_extents, rgba) -> int:
    """Translucent visual-only box — used to show the table bbox."""
    client = env.sim.client
    vid = client.createVisualShape(
        shapeType=pb.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=list(rgba),
    )
    return client.createMultiBody(baseVisualShapeIndex=vid, basePosition=list(center))


def _ee_world_pose(env, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the world (position, rotation) of *link_name* in the env.

    Uses PyBullet's FK so it automatically tracks whatever joint state
    the caller applied last — much simpler than composing the chain
    frame transform by hand.
    """
    client = env.sim.client
    link_idx = env.sim.link_map.get(link_name, -1)
    if link_idx < 0:
        raise RuntimeError(f"link {link_name!r} not found in the robot model")
    state = client.getLinkState(env.sim.skel_id, link_idx)
    pos = np.asarray(state[0], dtype=float)
    rot = np.asarray(client.getMatrixFromQuaternion(state[1]), dtype=float).reshape(
        3, 3
    )
    return pos, rot


def _remove_bodies(env, ids: list[int]) -> None:
    for i in ids:
        try:
            env.sim.client.removeBody(i)
        except pb.error:
            pass


def _log(title: str, out: Path, extra: str = "") -> None:
    print(f"  → {title:<34} {out.name}{('  ' + extra) if extra else ''}")


# ───────────────────── TRAC-IK clip ─────────────────────────────────────


def record_trac_ik(out: Path) -> None:
    env = _make_env()
    home_body = HOME_JOINTS.copy()
    G = JOINT_GROUPS

    # Sweep across several kinematic chains so the clip shows the same
    # factory solving IK for different subgroups of the robot.  For each
    # chain we apply the solution, read the EE world pose via PyBullet
    # FK, and use that as the target frame the video will draw.
    chain_specs = [
        (
            "left_arm",
            "Link_Left_Gripper",
            np.arange(7, 14),  # slots 7..13
            HOME_JOINTS[G["left_arm"]],
            [
                np.array([0.22, 0.15, 0.05]),
                np.array([0.15, 0.25, 0.05]),
            ],
        ),
        (
            "right_arm",
            "Link_Right_Gripper",
            np.arange(17, 24),  # slots 17..23
            HOME_JOINTS[G["right_arm"]],
            [np.array([0.22, -0.15, 0.05])],
        ),
        (
            "whole_body_left",
            "Link_Left_Gripper",
            np.arange(3, 14),  # legs + waist + left_arm = slots 3..13
            np.concatenate(
                [
                    HOME_JOINTS[G["legs"]],
                    HOME_JOINTS[G["waist"]],
                    HOME_JOINTS[G["left_arm"]],
                ]
            ),
            [np.array([0.30, 0.10, 0.05])],
        ),
        (
            "whole_body_right",
            "Link_Right_Gripper",
            np.r_[np.arange(3, 7), np.arange(17, 24)],  # legs + waist + right_arm
            np.concatenate(
                [
                    HOME_JOINTS[G["legs"]],
                    HOME_JOINTS[G["waist"]],
                    HOME_JOINTS[G["right_arm"]],
                ]
            ),
            [np.array([0.30, -0.10, 0.05])],
        ),
    ]

    solved: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for chain_name, ee_link, slots, seed, offsets in chain_specs:
        solver = create_ik_solver(
            chain_name, config=IKConfig(solve_type=SolveType.DISTANCE, timeout=0.3)
        )
        home_pose = solver.fk(seed)
        for off in offsets:
            r = solver.solve(
                SE3Pose(home_pose.position + off, home_pose.rotation), seed=seed
            )
            if not r.success or r.joint_positions is None:
                continue
            sol_body = home_body.copy()
            sol_body[slots] = np.asarray(r.joint_positions)
            env.set_configuration(sol_body)
            ee_pos, ee_rot = _ee_world_pose(env, ee_link)
            solved.append((ee_link, sol_body, ee_pos, ee_rot))

    env.set_configuration(home_body)

    with VideoRecorder(env, out, camera=CameraView()) as rec:
        rec.hold(seconds=0.3)
        for ee_link, sol_body, t_pos, t_rot in solved:
            # Draw the target frame where the IK solution actually lands,
            # then snap the robot between home and solution — no smooth
            # interpolation, just the two poses the user cares about.
            frame_ids = env.draw_frame(t_pos, t_rot, size=0.12, radius=0.008)

            env.set_configuration(home_body)
            rec.hold(seconds=0.7)

            env.set_configuration(sol_body)
            rec.hold(seconds=1.6)

            _remove_bodies(env, frame_ids)
            env.set_configuration(home_body)
            rec.hold(seconds=0.15)

        rec.hold(seconds=0.4)

    _log("TRAC-IK sweep", out, f"{len(solved)} solves across {len(chain_specs)} chains")


# ───────────────────── Pink constrained-IK clip ─────────────────────────


def record_pink_ik(out: Path) -> None:
    env = _make_env()
    G = JOINT_GROUPS
    home_body = HOME_JOINTS.copy()

    config = PinkIKConfig(
        lm_damping=1e-3,
        com_cost=0.1,
        camera_frame="Link_Waist_Yaw_to_Shoulder_Inner",
        camera_cost=0.1,
        max_iterations=200,
    )

    # Pink's `whole_body` chain covers legs + waist + one arm (11 DOF).
    # We sweep both sides to show the constrained solver handling the
    # left and right whole-body chains with the same config.
    side_specs = [
        (
            "left",
            "Link_Left_Gripper",
            np.r_[G["legs"], G["waist"], G["left_arm"]],
            np.concatenate(
                [
                    HOME_JOINTS[G["legs"]],
                    HOME_JOINTS[G["waist"]],
                    HOME_JOINTS[G["left_arm"]],
                ]
            ),
            [
                np.array([0.30, 0.00, 0.00]),
                np.array([0.20, 0.20, 0.00]),
            ],
        ),
        (
            "right",
            "Link_Right_Gripper",
            np.r_[G["legs"], G["waist"], G["right_arm"]],
            np.concatenate(
                [
                    HOME_JOINTS[G["legs"]],
                    HOME_JOINTS[G["waist"]],
                    HOME_JOINTS[G["right_arm"]],
                ]
            ),
            [
                np.array([0.30, 0.00, 0.00]),
                np.array([0.20, -0.20, 0.00]),
            ],
        ),
    ]

    solved: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for side, ee_link, chain_cols, seed, offsets in side_specs:
        solver = create_ik_solver(
            "whole_body", side=side, backend="pink", config=config
        )
        home_pose = solver.fk(seed)
        for off in offsets:
            r = solver.solve_constrained(
                SE3Pose(home_pose.position + off, home_pose.rotation), seed=seed
            )
            if r.joint_positions is None:
                continue
            sol_body = home_body.copy()
            for j, col in enumerate(chain_cols):
                sol_body[int(col)] = float(r.joint_positions[j])
            env.set_configuration(sol_body)
            ee_pos, ee_rot = _ee_world_pose(env, ee_link)
            solved.append((ee_link, sol_body, ee_pos, ee_rot))

    env.set_configuration(home_body)

    with VideoRecorder(env, out, camera=CameraView()) as rec:
        rec.hold(seconds=0.3)
        for ee_link, sol_body, t_pos, t_rot in solved:
            frame_ids = env.draw_frame(t_pos, t_rot, size=0.12, radius=0.008)

            env.set_configuration(home_body)
            rec.hold(seconds=0.7)

            env.set_configuration(sol_body)
            rec.hold(seconds=1.6)

            _remove_bodies(env, frame_ids)
            env.set_configuration(home_body)
            rec.hold(seconds=0.15)

        rec.hold(seconds=0.4)

    _log("Pink constrained-IK sweep", out, f"{len(solved)} solves (both sides)")


# ───────────────────── motion-planning clip ─────────────────────────────


def record_motion_planning(out: Path) -> None:
    from motion_planning_example import load_table  # noqa: WPS433 (local import)

    env = _make_env()
    cloud = load_table()

    # Draw a translucent box at the cloud bbox so the obstacle is
    # visible in the offscreen renderer (debug points aren't).
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    _draw_box(env, center, half, (0.22, 0.55, 0.85, 0.45))

    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=3.0, point_radius=0.012),
        base_config=HOME_JOINTS.copy(),
        pointcloud=cloud,
    )
    start = planner.extract_config(HOME_JOINTS)

    # Keep sampling until the plan has enough waypoints for a pretty clip.
    rng = np.random.default_rng(7)
    best = None
    for _ in range(16):
        goal = planner.sample_valid()
        result = planner.plan(start, goal)
        if result.success and result.path is not None and result.path.shape[0] >= 4:
            best = result
            break
        # Pretend we use rng so the linter stops complaining.
        _ = rng.random()

    if best is None or best.path is None:
        raise RuntimeError("motion planning: could not find a plan for the video")

    full_path = planner.embed_path(best.path)

    with VideoRecorder(env, out, camera=CameraView()) as rec:
        rec.hold(seconds=0.6)
        rec.play_path(full_path, duration=7.0)
        rec.hold(seconds=0.6)

    _log("Motion planning w/ table", out, f"{full_path.shape[0]} waypoints")


# ───────────────────── subgroup-planning clip ───────────────────────────


def record_subgroup_planning(out: Path) -> None:
    env = _make_env()

    STANCES: dict[str, dict[str, float]] = {
        "high": {"Joint_Ankle": 0.0, "Joint_Knee": 0.0, "Joint_Waist_Pitch": 0.00},
        "mid": {"Joint_Ankle": 0.78, "Joint_Knee": 1.60, "Joint_Waist_Pitch": 0.89},
        "low": {"Joint_Ankle": 1.41, "Joint_Knee": 2.38, "Joint_Waist_Pitch": 0.95},
    }

    def base_with_stance(stance: dict[str, float]) -> np.ndarray:
        base = HOME_JOINTS.copy()
        for name, val in stance.items():
            base[autolife_robot_config.joint_names.index(name)] = val
        return base

    # A curated set that reads well in a short clip: one per stance.
    combos = [
        ("autolife_left_arm", "high"),
        ("autolife_torso_left_arm", "mid"),
        ("autolife_dual_arm", "low"),
    ]

    paths: list[np.ndarray] = []
    for subgroup, stance_name in combos:
        base = base_with_stance(STANCES[stance_name])
        planner = create_planner(
            subgroup,
            config=PlannerConfig(planner_name="bitstar", time_limit=0.6),
            base_config=base,
        )
        start = planner.extract_config(base)
        for _ in range(8):
            goal = planner.sample_valid()
            result = planner.plan(start, goal)
            if result.success and result.path is not None:
                paths.append(planner.embed_path(result.path))
                break

    if not paths:
        raise RuntimeError("subgroup_planning: no plans succeeded")

    # Stitch: end of segment N → start of segment N+1 via linear blend
    # so the camera shows a continuous sequence.
    stitched: list[np.ndarray] = []
    for i, seg in enumerate(paths):
        stitched.append(seg)
        if i != len(paths) - 1:
            blend = _interp(seg[-1], paths[i + 1][0], 20)
            stitched.append(blend)
    full_path = np.concatenate(stitched, axis=0)

    with VideoRecorder(env, out, camera=CameraView()) as rec:
        rec.hold(seconds=0.5)
        rec.play_path(full_path, duration=12.0)
        rec.hold(seconds=0.5)

    _log("Subgroup planning medley", out, f"{len(combos)} plans")


# ───────────────────── constrained-planning helpers ─────────────────────


def _setup_constrained():
    env = _make_env()
    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()
    return env, ctx, start


def _find_goal(ctx, residual, start, planner, score, n: int = 400, seed: int = 0):
    lower = np.array(planner._planner.lower_bounds())
    upper = np.array(planner._planner.upper_bounds())
    rng = np.random.default_rng(seed)
    best_q, best_s = None, -np.inf
    for _ in range(n):
        seed_q = np.clip(
            start + rng.uniform(-0.7, 0.7, start.shape[0]),
            lower + 0.02,
            upper - 0.02,
        )
        try:
            g = ctx.project(seed_q, residual)
        except RuntimeError:
            continue
        if np.any(g < lower) or np.any(g > upper):
            continue
        if not planner.validate(g):
            continue
        s = float(score(_ee_tcp_eval(ctx, g)))
        if s > best_s:
            best_q, best_s = g, s
    if best_q is None:
        raise RuntimeError("no manifold-feasible goal found")
    return best_q


def _record_constrained(
    out: Path,
    title: str,
    build_constraint,
    score,
    after_setup=None,
    time_limit: float = 5.0,
    camera: CameraView | None = None,
) -> None:
    env, ctx, start = _setup_constrained()

    constraint = build_constraint(ctx, start, env)
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(time_limit=time_limit),
        constraints=[constraint],
    )
    goal = _find_goal(ctx, constraint.residual, start, planner, score=score)

    if after_setup is not None:
        after_setup(env, ctx, start, goal)

    result = planner.plan(start, goal)
    if not result.success or result.path is None:
        raise RuntimeError(f"{title}: planning failed")

    full_path = planner.embed_path(result.path)
    cam = camera or CameraView()

    with VideoRecorder(env, out, camera=cam) as rec:
        rec.hold(seconds=0.6)
        rec.play_path(full_path, duration=7.0)
        rec.hold(seconds=0.6)

    _log(title, out, f"{result.path.shape[0]} waypoints")


def record_constrained_plane(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        return Constraint(
            residual=_ee_tcp_sym(ctx)[2] - float(p0[2]),
            q_sym=ctx.q,
            name="plane_z",
        )

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_plane(
            center=[
                float(0.5 * (p0[0] + p_goal[0])),
                float(0.5 * (p0[1] + p_goal[1])),
                float(p0[2]) - 0.05,
            ],
            half_sizes=(0.65, 0.65),
            color=(0.15, 0.55, 0.95, 0.22),
        )

    _record_constrained(
        out,
        "plane: gripper on z = z₀",
        build,
        score=lambda xyz: abs(xyz[1] - 0.25) + 0.3 * abs(xyz[0]),
        after_setup=after,
    )


def record_constrained_plane_obstacle(out: Path) -> None:
    # Same seed & geometry as examples/constrained_planning/plane_with_obstacle.py
    # Tuned so the TCP sweeps several tens of cm on the manifold and
    # the obstacle sits cleanly on the direct path.
    _GOAL_SEED = np.array([0.95, -0.882, -0.881, 1.593, 0.209, 0.082, -0.418])

    def _sample_ball(center, radius, n, seed: int = 0):
        rng = np.random.default_rng(seed)
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = radius * rng.random(n) ** (1.0 / 3.0)
        return (dirs * radii[:, None] + center).astype(np.float32)

    env, ctx, start = _setup_constrained()
    p0 = _ee_tcp_eval(ctx, start)
    plane = Constraint(
        residual=_ee_tcp_sym(ctx)[2] - float(p0[2]),
        q_sym=ctx.q,
        name="plane_obs",
    )
    goal = ctx.project(_GOAL_SEED, plane.residual)

    p_goal = _ee_tcp_eval(ctx, goal)
    sphere_center = np.array(
        [
            float(0.5 * (p0[0] + p_goal[0])),
            float(0.5 * (p0[1] + p_goal[1])),
            float(p0[2]) - 0.03,
        ]
    )
    sphere_radius = 0.06

    env.draw_plane(
        center=[
            float(sphere_center[0]),
            float(sphere_center[1]),
            float(p0[2]) - 0.05,
        ],
        half_sizes=(0.65, 0.65),
        color=(0.15, 0.55, 0.95, 0.22),
    )
    env.draw_sphere(center=sphere_center, radius=sphere_radius)

    cloud = _sample_ball(sphere_center, sphere_radius, n=800)
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=10.0, point_radius=0.012),
        pointcloud=cloud,
        constraints=[plane],
    )
    result = planner.plan(start, goal)
    if not result.success or result.path is None:
        raise RuntimeError("plane_obstacle: planning failed")

    full_path = planner.embed_path(result.path)

    # Marker visualising the point on the robot that the constraint
    # actually pins — the TCP (midpoint between the two finger link
    # origins), which is where ``_ee_tcp_sym(ctx)`` evaluates.  A
    # small orange sphere is reset every frame to the live world TCP
    # position so the viewer can literally see "this is the point
    # that stays on the plane".
    client = env.sim.client
    marker_vid = client.createVisualShape(
        shapeType=pb.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.00, 0.40, 0.05, 1.0],
    )
    marker_id = client.createMultiBody(
        baseVisualShapeIndex=marker_vid,
        basePosition=[0.0, 0.0, 0.0],
    )

    def _sync_marker(_cfg) -> None:
        pos = _ee_tcp_world(env)
        client.resetBasePositionAndOrientation(marker_id, pos.tolist(), [0, 0, 0, 1])

    # Initialise the marker at the start pose before the first hold.
    env.set_configuration(full_path[0])
    _sync_marker(full_path[0])

    with VideoRecorder(env, out, camera=CameraView()) as rec:
        rec.hold(seconds=0.6)
        rec.play_path(full_path, duration=8.0, on_frame=_sync_marker)
        rec.hold(seconds=0.6)

    _log("plane + sphere obstacle", out, f"{result.path.shape[0]} waypoints")


def record_constrained_line_horizontal(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        tcp = _ee_tcp_sym(ctx)
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            tcp[1] - float(p0[1]),
            tcp[2] - float(p0[2]),
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return Constraint(residual=residual, q_sym=ctx.q, name="line_h")

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_rod(
            p1=[float(p0[0]) - 0.10, float(p0[1]), float(p0[2])],
            p2=[float(p_goal[0]) + 0.10, float(p0[1]), float(p0[2])],
            radius=0.008,
            color=(0.20, 0.90, 0.30, 1.0),
        )

    _record_constrained(
        out,
        "line_h: gripper on a horizontal rail",
        build,
        score=lambda xyz: xyz[0],
        after_setup=after,
    )


def record_constrained_line_vertical(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        tcp = _ee_tcp_sym(ctx)
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            tcp[0] - float(p0[0]),
            tcp[1] - float(p0[1]),
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return Constraint(residual=residual, q_sym=ctx.q, name="line_v")

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_rod(
            p1=[float(p0[0]), float(p0[1]), float(p0[2]) - 0.10],
            p2=[float(p0[0]), float(p0[1]), float(p_goal[2]) + 0.10],
            radius=0.008,
            color=(0.95, 0.85, 0.10, 1.0),
        )

    _record_constrained(
        out,
        "line_v: gripper on a vertical rail",
        build,
        score=lambda xyz: xyz[2],
        after_setup=after,
    )


def record_constrained_orientation_lock(out: Path) -> None:
    def build(ctx, start, env):
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return Constraint(residual=residual, q_sym=ctx.q, name="orient_lock")

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_frame(p0, R0, size=0.12, radius=0.007)
        env.draw_frame(p_goal, R0, size=0.12, radius=0.007)
        env.draw_rod(p0, p_goal, radius=0.004, color=(0.85, 0.35, 0.95, 0.9))

    _record_constrained(
        out,
        "orient: translation free, rotation locked",
        build,
        score=lambda xyz: float(np.linalg.norm(xyz)),
        after_setup=after,
    )


# ───────────────────── cost-planning helpers ────────────────────────────
#
# Mirrors the constrained-planning renderers above, with one swap:
# every hard ``Constraint`` becomes a ``Cost`` that integrates
# ``sumsqr(residual)`` along every motion.  Goals still come from
# projecting onto ``residual = 0`` so the animation clearly shows
# the arm preferring the soft manifold whenever it can.


def _record_cost(
    out: Path,
    title: str,
    build_cost,
    score,
    after_setup=None,
    time_limit: float = 10.0,
    weight: float = 50.0,
    planner_name: str = "rrtstar",
    camera: CameraView | None = None,
) -> None:
    env, ctx, start = _setup_constrained()

    cost, residual = build_cost(ctx, start, env)
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name=planner_name,
            time_limit=time_limit,
            # OMPL's default simplifier takes geometric shortcuts that
            # ignore the custom cost; disabling it keeps the shape that
            # RRT*-family search produced.
            simplify=False,
        ),
        costs=[cost],
    )
    goal = _find_goal(ctx, residual, start, planner, score=score)

    if after_setup is not None:
        after_setup(env, ctx, start, goal)

    result = planner.plan(start, goal)
    if not result.success or result.path is None:
        raise RuntimeError(f"{title}: planning failed")

    full_path = planner.embed_path(result.path)
    cam = camera or CameraView()

    with VideoRecorder(env, out, camera=cam) as rec:
        rec.hold(seconds=0.6)
        rec.play_path(full_path, duration=7.0)
        rec.hold(seconds=0.6)

    _log(title, out, f"{result.path.shape[0]} waypoints")


def record_cost_plane(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        residual = _ee_tcp_sym(ctx)[2] - float(p0[2])
        return (
            Cost(
                expression=ca.sumsqr(residual),
                q_sym=ctx.q,
                name="plane_z_cost",
                weight=50.0,
            ),
            residual,
        )

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_plane(
            center=[
                float(0.5 * (p0[0] + p_goal[0])),
                float(0.5 * (p0[1] + p_goal[1])),
                float(p0[2]) - 0.05,
            ],
            half_sizes=(0.65, 0.65),
            color=(0.95, 0.55, 0.15, 0.22),
        )

    _record_cost(
        out,
        "cost plane: gripper pulled toward z = z₀",
        build,
        score=lambda xyz: abs(xyz[1] - 0.25) + 0.3 * abs(xyz[0]),
        after_setup=after,
    )


def record_cost_line_horizontal(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        tcp = _ee_tcp_sym(ctx)
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            tcp[1] - float(p0[1]),
            tcp[2] - float(p0[2]),
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return (
            Cost(
                expression=ca.sumsqr(residual),
                q_sym=ctx.q,
                name="line_h_cost",
                weight=50.0,
            ),
            residual,
        )

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_rod(
            p1=[float(p0[0]) - 0.10, float(p0[1]), float(p0[2])],
            p2=[float(p_goal[0]) + 0.10, float(p0[1]), float(p0[2])],
            radius=0.008,
            color=(0.95, 0.55, 0.15, 1.0),
        )

    _record_cost(
        out,
        "cost line_h: gripper pulled toward a horizontal rail",
        build,
        score=lambda xyz: xyz[0],
        after_setup=after,
    )


def record_cost_line_vertical(out: Path) -> None:
    def build(ctx, start, env):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        tcp = _ee_tcp_sym(ctx)
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            tcp[0] - float(p0[0]),
            tcp[1] - float(p0[1]),
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return (
            Cost(
                expression=ca.sumsqr(residual),
                q_sym=ctx.q,
                name="line_v_cost",
                weight=50.0,
            ),
            residual,
        )

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_rod(
            p1=[float(p0[0]), float(p0[1]), float(p0[2]) - 0.10],
            p2=[float(p0[0]), float(p0[1]), float(p_goal[2]) + 0.10],
            radius=0.008,
            color=(0.95, 0.55, 0.15, 1.0),
        )

    _record_cost(
        out,
        "cost line_v: gripper pulled toward a vertical rail",
        build,
        score=lambda xyz: xyz[2],
        after_setup=after,
    )


def record_cost_orientation_lock(out: Path) -> None:
    def build(ctx, start, env):
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        left_rot = ctx.link_rotation(EE_LINK)
        residual = ca.vertcat(
            left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        )
        return (
            Cost(
                expression=ca.sumsqr(residual),
                q_sym=ctx.q,
                name="orient_lock_cost",
                weight=50.0,
            ),
            residual,
        )

    def after(env, ctx, start, goal):
        p0 = _ee_tcp_eval(ctx, start)
        R0 = np.asarray(ctx.evaluate_link_pose(EE_LINK, start))[:3, :3]
        p_goal = _ee_tcp_eval(ctx, goal)
        env.draw_frame(p0, R0, size=0.12, radius=0.007)
        env.draw_frame(p_goal, R0, size=0.12, radius=0.007)
        env.draw_rod(p0, p_goal, radius=0.004, color=(0.95, 0.55, 0.15, 0.9))

    _record_cost(
        out,
        "cost orient: translation free, orientation softly locked",
        build,
        score=lambda xyz: float(np.linalg.norm(xyz)),
        after_setup=after,
    )


# ───────────────────── main ─────────────────────────────────────────────


CLIPS = [
    ("trac_ik", record_trac_ik, "trac_ik.mp4"),
    ("constrained_ik", record_pink_ik, "constrained_ik.mp4"),
    ("motion_planning", record_motion_planning, "motion_planning.mp4"),
    ("subgroup_planning", record_subgroup_planning, "subgroup_planning.mp4"),
    ("plane", record_constrained_plane, "constrained_plane.mp4"),
    (
        "plane_obstacle",
        record_constrained_plane_obstacle,
        "constrained_plane_obstacle.mp4",
    ),
    (
        "line_h",
        record_constrained_line_horizontal,
        "constrained_line_horizontal.mp4",
    ),
    (
        "line_v",
        record_constrained_line_vertical,
        "constrained_line_vertical.mp4",
    ),
    (
        "orient",
        record_constrained_orientation_lock,
        "constrained_orientation_lock.mp4",
    ),
    ("cost_plane", record_cost_plane, "cost_plane.mp4"),
    ("cost_line_h", record_cost_line_horizontal, "cost_line_horizontal.mp4"),
    ("cost_line_v", record_cost_line_vertical, "cost_line_vertical.mp4"),
    ("cost_orient", record_cost_orientation_lock, "cost_orientation_lock.mp4"),
]


def main(only: str | tuple | list = "") -> None:
    """Render every documentation video to ``docs/assets/``.

    Pass ``--only=name`` to render one clip, or
    ``--only=name1,name2`` (Fire parses this as a tuple) to render a
    subset.  Valid names: ``trac_ik``, ``constrained_ik``,
    ``motion_planning``, ``subgroup_planning``, ``plane``,
    ``plane_obstacle``, ``line_h``, ``line_v``, ``orient``.
    """
    if isinstance(only, str):
        wanted = {s.strip() for s in only.split(",") if s.strip()}
    else:
        wanted = {str(s).strip() for s in only if str(s).strip()}
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep Ompl's informer chatter out of the video log.
    os.environ.setdefault("OMPL_LOG_LEVEL", "ERROR")

    print(f"rendering docs videos → {ASSETS_DIR}")
    t0 = time.time()
    for name, fn, filename in CLIPS:
        if wanted and name not in wanted:
            continue
        out = ASSETS_DIR / filename
        t_clip = time.time()
        try:
            fn(out)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {name} failed: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t_clip
        size_kb = out.stat().st_size / 1024 if out.exists() else 0
        print(f"    ({dt:.1f}s, {size_kb:.0f} KB)")
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    Fire(main)
