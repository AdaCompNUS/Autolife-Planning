"""Long-horizon pick-and-place inside the full RLS-env room.

A single end-to-end showcase of the project's planning stack:

* **Four subgroup planners** combined in one sequence:
  ``autolife_height`` (squat/stand), ``autolife_body`` (whole upper
  body with leg-pin constraint), ``autolife_torso_left_arm``
  (9-DOF arm), and ``autolife_base`` (navigation).
* **CasADi constraints** — the under-table pick pins the ankle +
  knee via a compiled holonomic constraint while the arm plans on
  the remaining 19 DOF; every pregrasp→grasp sweep uses a
  2-equation straight-line manifold; the kitchen-counter carry
  phase uses a horizontal-hold orientation constraint.
* **Collision avoidance** — every planner gets the full 151 k-point
  cloud built from all seven ``pcd/*.ply`` scans.
* **Hardcoded grasp configs** — each grasp/pregrasp/place pose is a
  pre-solved 24-DOF configuration so the demo is 100 % deterministic.

Storyline:

    1a. squat down  (autolife_height, 3 DOF)
    1b. pick apple from under the table  (autolife_body, 21 DOF + leg pin)
    1c. stand up  (autolife_height)
    2.  place apple on table top  (autolife_torso_left_arm, 9 DOF)
    3.  navigate to kitchen counter  (autolife_base)
    4.  pick bowl from kitchen counter  (autolife_torso_left_arm)
        → low-hanging beam spawns over the kitchen→workstation corridor
    5.  drive to coffee table, auto-squatting under the beam  (custom
        6-DOF subgroup ``base + height-chain`` planned in one shot
        with RRT-Connect on the projected state space; hard constraints
        ``knee = 2·ankle``, ``waist_pitch = ankle`` and
        ``yaw = current_yaw``; under the leg coupling the upper body
        stays vertical so the bowl's full SO(3) pose is preserved for
        free; path-length simplification trims the squat to the minimum
        the obstacle actually demands)
    6.  place bowl on coffee table  (autolife_torso_left_arm)
    7.  navigate to sofa  (autolife_base, 3 DOF)
    8.  pick bottle from sofa  (autolife_torso_left_arm)
    9.  navigate to coffee table  (autolife_base)
    10. place bottle on coffee table  (autolife_torso_left_arm)

Usage::

    pixi run python examples/demos/rls_pick_place.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pybullet as pb
import trimesh
from fire import Fire

from autolife_planning.autolife import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import Constraint, Cost, SymbolicContext, create_planner
from autolife_planning.types import PlannerConfig

# ── Custom subgroup: base + height chain ──────────────────────────
# Lets a single planner vary the base (x, y, yaw) AND the height
# chain (ankle, knee, waist_pitch) at the same time — a 6-DOF active
# set.  waist_yaw and the right arm are deliberately *not* active:
# they stay at their ``base_config`` values (the bowl-grasp pose)
# throughout the plan.  The bowl's world-frame orientation is held
# level by a hard horizontal-hold constraint on the right gripper
# (``R_grip[0,2] = R_grip[1,2] = 0``), which — since the arm and
# waist_yaw are frozen — reduces to an implicit coupling that forces
# ``waist_pitch`` to compensate ``ankle`` at every manifold sample.
DRIVE_SQUAT_SUBGROUP = "autolife_drive_squat"
PLANNING_SUBGROUPS[DRIVE_SQUAT_SUBGROUP] = {
    "dof": 6,
    "joints": [
        "Joint_Virtual_X",
        "Joint_Virtual_Y",
        "Joint_Virtual_Theta",
        "Joint_Ankle",
        "Joint_Knee",
        "Joint_Waist_Pitch",
    ],
}
DRIVE_SQUAT_IDX = np.array([0, 1, 2, 3, 4, 5])
DRIVE_SQUAT_YAW_IDX = 2  # inside the 6-element active vector
DRIVE_SQUAT_ANKLE_IDX = 3
DRIVE_SQUAT_KNEE_IDX = 4
DRIVE_SQUAT_WAIST_PITCH_IDX = 5

# ── Subgroups ──────────────────────────────────────────────────────
BASE_SUBGROUP = "autolife_base"  # 3 DOF: virtual x/y/yaw
HEIGHT_SUBGROUP = "autolife_height"  # 3 DOF: ankle, knee, waist_pitch
ARM_SUBGROUP = "autolife_torso_left_arm"  # 9 DOF: waist + left arm
BODY_SUBGROUP = "autolife_body"  # 21 DOF: legs + waist + arms + neck

TORSO_ARM_IDX = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13])
RARM_SUBGROUP = "autolife_torso_right_arm"  # 9 DOF: waist + right arm
TORSO_RARM_IDX = np.array([5, 6, 17, 18, 19, 20, 21, 22, 23])
RIGHT_GRIPPER_LINK = "Link_Right_Gripper"
FINGER_MIDPOINT_IN_RIGHT_GRIPPER = np.array([0.031, -0.064, 0.0])
HEIGHT_IDX = np.array([3, 4, 5])
BODY_IDX = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
)
BODY_ANKLE_IDX = 0  # inside the 21-element active vector
BODY_KNEE_IDX = 1

GRIPPER_LINK = "Link_Left_Gripper"

# ── Asset paths ────────────────────────────────────────────────────
RLS_ROOT = "assets/envs/rls_env"
MESH_DIR = f"{RLS_ROOT}/meshes"
PCD_DIR = f"{RLS_ROOT}/pcd"
SCENE_PROPS = [
    ("rls_2", "rls_2"),
    ("open_kitchen", "open_kitchen"),
    ("wall", "wall"),
    ("workstation", "workstation"),
    ("table", "table"),
    ("sofa", "sofa"),
    ("tea_table", "coffee_table"),
]

# ── Robot base poses ──────────────────────────────────────────────
BASE_SQUAT_EAST = np.array([-1.10, 1.67, np.pi])  # for squat + under-table pick
BASE_PLACE_EAST = np.array([-1.30, 1.67, np.pi])  # for placing on the table top
BASE_NEAR_KITCHEN = np.array([-3.80, -2.40, -np.pi / 2])  # beside kitchen counter
BASE_NEAR_SOFA = np.array([2.00, 1.30, -np.pi / 2])
BASE_NEAR_COFFEE = np.array([3.60, 1.30, -np.pi / 2])
STANDING_STANCE = np.array([0.0, 0.0, 0.0])
# Sit-down pose: ankle=-1.20, knee=-2.40 (lower leg folds backward,
# opposite to human), waist_pitch=-1.20 (torso compensates to stay
# upright).
SQUAT_STANCE = np.array([-1.20, -2.40, -1.20])

# ── Graspable positions + approach vectors ────────────────────────
APPLE_MESH_INIT = np.array([-1.50, 1.67, 0.15])  # on the floor near the table
APPLE_MESH_PLACE = np.array([-2.05, 1.67, 0.75])
APPLE_APPROACH = np.array([-1.0, 0.0, 0.0])
APPLE_GRASP_Z = 0.13

BOWL_MESH_INIT = np.array([-4.00, -2.93, 0.89])  # on kitchen counter
BOWL_MESH_PLACE = np.array([3.55, 0.45, 0.58])  # on coffee table
BOWL_APPROACH = np.array([0.0, -1.0, 0.0])

BOTTLE_MESH_INIT = np.array([1.95, 0.30, 0.69])
BOTTLE_MESH_PLACE = np.array([3.60, 0.30, 0.58])
BOTTLE_APPROACH = np.array([0.0, -1.0, 0.0])
BOTTLE_GRASP_Z = 0.14

PREGRASP_OFFSET = 0.12
PREGRASP_OFFSET_UNDER_TABLE = 0.05

# ── Hardcoded 24-DOF grasp/pregrasp configs ───────────────────────
# Pre-solved via TRAC-IK on the whole_body_base_left chain and
# validated against the scene pointcloud.  Each config is the FULL
# 24-DOF body state (base + legs + waist + arms + neck).
# fmt: off
_NH = [0.0, 0.0, 0.0, -0.7, 0.14, -0.09, -2.31, -0.04, -0.4, 0.0]

APPLE_GRASP_FULL = np.array(
    [-1.1, 1.67, 3.14159, -1.2, -2.4, -0.14577, -1.31109,
     0.40492, -0.1654, -0.10294, 2.02168, -0.00743,
     -1.28599, -0.85198] + _NH)
APPLE_PREGRASP_FULL = np.array(
    [-1.1, 1.67, 3.14159, -1.2, -2.4, -0.09358, -1.30795,
     0.43711, -0.06845, -0.04382, 1.99664, 0.00559,
     -1.29513, -0.87619] + _NH)
APPLE_PLACE_FULL = np.array(
    [-1.3, 1.67, 3.14159, 0.0, 0.0, 0.54695, -0.94467,
     -0.39819, -0.36978, -0.32065, 1.66405, 0.13771,
     -0.5133, 0.13768] + _NH)
APPLE_PREPLACE_FULL = np.array(
    [-1.3, 1.67, 3.14159, 0.0, 0.0, 0.50202, -0.84614,
     -0.21316, -0.11009, -0.19886, 1.7565, 0.22301,
     -0.59207, 0.00636] + _NH)
BOTTLE_GRASP_FULL = np.array(
    [2.0, 1.3, -1.5708, 0.0, 0.0, 0.7048, -1.4169,
     -0.11159, -1.38957, -0.47527, 1.08013, -0.72665,
     -0.71251, -0.15198] + _NH)
BOTTLE_PREGRASP_FULL = np.array(
    [2.0, 1.3, -1.5708, 0.0, 0.0, 0.63199, -1.31205,
     -0.30518, -0.83376, -0.62307, 1.42075, -0.09389,
     -0.54424, 0.07077] + _NH)
BOTTLE_PLACE_FULL = np.array(
    [3.6, 1.3, -1.5708, 0.0, 0.0, 0.926, -1.38697,
     -0.32474, -1.45883, -0.5743, 1.09816, -0.4848,
     -0.48822, -0.1056] + _NH)
BOTTLE_PREPLACE_FULL = np.array(
    [3.6, 1.3, -1.5708, 0.0, 0.0, 0.90982, -1.23524,
     -0.56996, -0.89735, -0.65942, 1.53732, 0.04446,
     -0.26316, 0.08853] + _NH)
# Bowl on kitchen counter – right arm, base at [-3.8, -2.4, -pi/2].
# All four configs satisfy the FULL right-gripper rotation lock R = I
# (X axis world-X, Y axis world-Y, Z axis world-Z = palm up, fingers
# pointing south = forward).  Connected via line-constraint manifold.
BOWL_GRASP_FULL = np.array([
    -3.8, -2.4, -1.5708, 0.0, 0.0, -0.03787, -0.00501,
    0.7, -0.14, -0.09, 2.31, 0.04, -0.4, 0.0,
    0.0, 0.0, 0.0,
    -2.31808, 2.86290, 0.96723, -0.67625, 2.17094, 0.28252, -1.08729,
])
BOWL_PREGRASP_FULL = np.array([
    -3.8, -2.4, -1.5708, 0.0, 0.0, -0.30794, -0.21616,
    0.7, -0.14, -0.09, 2.31, 0.04, -0.4, 0.0,
    0.0, 0.0, 0.0,
    -2.45845, 2.92047, 0.84476, -0.85045, 2.12556, 0.38272, -1.06612,
])
# Bowl placement on coffee table – right arm, base at [3.6, 1.3, -pi/2].
BOWL_PLACE_FULL = np.array([
    3.6, 1.3, -1.5708, 0.0, 0.0, 0.96932, 1.09465,
    0.7, -0.14, -0.09, 2.31, 0.04, -0.4, 0.0,
    0.0, 0.0, 0.0,
    1.18665, 0.62293, 1.34621, -1.49030, -0.73090, -0.80767, -0.36327,
])
BOWL_PREPLACE_FULL = np.array([
    3.6, 1.3, -1.5708, 0.0, 0.0, 0.93529, 0.99476,
    0.7, -0.14, -0.09, 2.31, 0.04, -0.4, 0.0,
    0.0, 0.0, 0.0,
    0.94470, 0.32021, 1.51090, -1.68589, -0.63056, -0.84806, -0.47318,
])

# Right-gripper rotation lock target: identity (palm up, fingers south).
BOWL_HOLD_ROT = np.eye(3)
# fmt: on

# ── Low-hanging beam (activated AFTER the bowl is grasped) ────────
# A horizontal plank suspended over the open corridor between the
# kitchen counter and the workstation.  The robot, carrying the bowl
# with a locked horizontal pose, must duck under it by bending at the
# waist while the ankle + knee remain pinned (leg constraint) — and a
# soft cost encourages the upper body to stay as straight as possible,
# so the waist only bends as much as is needed to clear the plank.
BEAM_CENTER = np.array([-1.00, 0.00, 1.50])
BEAM_HALF_EXTENTS = np.array(
    [2.50, 0.20, 0.04]
)  # spans the entire free corridor in X (wall-to-wall)

ARM_FREE_TIME = 4.0
ARM_LINE_TIME = 8.0
BASE_NAV_TIME = 6.0


# ── Scene setup ────────────────────────────────────────────────────


def load_room_meshes(env: PyBulletEnv) -> None:
    for mesh_name, _ in SCENE_PROPS:
        env.add_mesh(
            os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj"),
            position=np.zeros(3),
        )


def load_room_pointcloud(stride: int = 1) -> np.ndarray:
    chunks = []
    for _, pcd_name in SCENE_PROPS:
        pc = trimesh.load(os.path.abspath(f"{PCD_DIR}/{pcd_name}.ply"))
        v = np.asarray(pc.vertices, dtype=np.float32)  # type: ignore[union-attr]
        if stride > 1:
            v = v[::stride]
        chunks.append(v)
    return np.concatenate(chunks, axis=0)


def place_graspable(env: PyBulletEnv, name: str, xyz: np.ndarray) -> int:
    return env.add_mesh(
        os.path.abspath(f"{MESH_DIR}/{name}/{name}.obj"),
        position=np.asarray(xyz, dtype=float),
    )


# ── Constraint helpers ─────────────────────────────────────────────


def _sanitize(s: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _project_and_clamp(ctx, residual, q, lo, hi, iters=5):
    for _ in range(iters):
        q = ctx.project(q, residual)
        q = np.clip(q, lo + 1e-5, hi - 1e-5)
    return q


def make_line_constraint(ctx, p_from, p_to, name, grip=GRIPPER_LINK):
    p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
    d = p_to - p_from
    d /= float(np.linalg.norm(d))
    seed = np.array([1, 0, 0.0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0.0])
    u = np.cross(d, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    gripper = ctx.link_translation(grip)
    diff = gripper - ca.DM(p_from.tolist())
    res = ca.vertcat(ca.dot(diff, ca.DM(u.tolist())), ca.dot(diff, ca.DM(v.tolist())))
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def make_leg_ratio(ctx, name):
    """Couple the legs as ``knee = 2 · ankle`` — single holonomic
    equation that keeps the foot flat during a squat without pinning
    either joint to an absolute value.
    """
    res = ctx.q[BODY_KNEE_IDX] - 2 * ctx.q[BODY_ANKLE_IDX]
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def make_horizontal_hold(ctx, name, grip=GRIPPER_LINK):
    """Constrain the gripper Z-axis to be vertical (object stays level).

    Returns a 2-equation constraint:  R_gripper[0,2] = 0, R_gripper[1,2] = 0.
    """
    R = ctx.link_rotation(grip)
    res = ca.vertcat(R[0, 2], R[1, 2])
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def grasp_rotation_from_approach(approach):
    a = np.asarray(approach, float)
    a /= float(np.linalg.norm(a))
    y = -a
    z = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1, 0, 0.0])
    z = z - float(np.dot(z, y)) * y
    z /= float(np.linalg.norm(z))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


FINGER_MIDPOINT_IN_GRIPPER = np.array([-0.031, -0.064, 0.0])


def grasp_line_endpoints(finger_xyz, approach, pregrasp_offset=PREGRASP_OFFSET):
    R = grasp_rotation_from_approach(approach)
    a = np.asarray(approach, float)
    a /= float(np.linalg.norm(a))
    grasp_pos = np.asarray(finger_xyz, float) - R @ FINGER_MIDPOINT_IN_GRIPPER
    pregrasp_pos = grasp_pos - pregrasp_offset * a
    return grasp_pos, pregrasp_pos


def fk_line_endpoints(
    grasp_full, pregrasp_full, subgroup, active_idx, grip=GRIPPER_LINK
):
    """Compute the gripper link positions from the PLANNER's own FK.

    The IK solver and SymbolicContext use different URDFs, so their FK
    differs.  This function ensures the line constraint endpoints match
    the planner's FK — the hardcoded configs already place the gripper
    at these positions by construction.
    """
    ctx = SymbolicContext(subgroup, base_config=grasp_full)
    gp = np.asarray(ctx.evaluate_link_pose(grip, grasp_full[active_idx]))[:3, 3]
    pp = np.asarray(ctx.evaluate_link_pose(grip, pregrasp_full[active_idx]))[:3, 3]
    return gp, pp


# ── Planning wrappers ──────────────────────────────────────────────


def _report(label, result):
    n = result.path.shape[0] if result.path is not None else 0
    ms = result.planning_time_ns / 1e6
    print(
        f"  [{label}] {result.status.value}: {n} wp in {ms:.0f} ms"
        + (f", cost {result.path_cost:.2f}" if result.success else "")
    )


def _use(planner, subgroup, current_full, constraints=None):
    """Switch the shared planner to a subgroup and optionally set constraints."""
    planner.set_subgroup(subgroup, base_config=current_full)
    if constraints:
        planner.set_constraints(constraints)


def _plan(planner, start, goal, label, time_limit=ARM_FREE_TIME):
    """Plan, report, assert success, and return the embedded full-DOF path."""
    r = planner.plan(start, goal, time_limit=time_limit)
    _report(label, r)
    assert r.success and r.path is not None, f"{label}: {r.status.value}"
    return planner.embed_path(r.path)


def _bounds(planner):
    return np.array(planner._planner.lower_bounds()), np.array(
        planner._planner.upper_bounds()
    )


def plan_arm_free(planner, current_full, goal_full, label):
    _use(planner, ARM_SUBGROUP, current_full)
    return _plan(planner, current_full[TORSO_ARM_IDX], goal_full[TORSO_ARM_IDX], label)


def plan_arm_line(planner, current_full, goal_full, p_from, p_to, label):
    ctx = SymbolicContext(ARM_SUBGROUP, base_config=current_full)
    c = make_line_constraint(ctx, p_from, p_to, f"line_{_sanitize(label)}")
    _use(planner, ARM_SUBGROUP, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(
        ctx, c.residual, current_full[TORSO_ARM_IDX].copy(), lo, hi
    )
    goal = _project_and_clamp(ctx, c.residual, goal_full[TORSO_ARM_IDX].copy(), lo, hi)
    return _plan(planner, start, goal, label, ARM_LINE_TIME)


def plan_rarm_free(planner, current_full, goal_full, label):
    _use(planner, RARM_SUBGROUP, current_full)
    return _plan(
        planner, current_full[TORSO_RARM_IDX], goal_full[TORSO_RARM_IDX], label
    )


def plan_rarm_line(planner, current_full, goal_full, p_from, p_to, label):
    ctx = SymbolicContext(RARM_SUBGROUP, base_config=current_full)
    c = make_line_constraint(
        ctx, p_from, p_to, f"line_{_sanitize(label)}", grip=RIGHT_GRIPPER_LINK
    )
    _use(planner, RARM_SUBGROUP, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(
        ctx, c.residual, current_full[TORSO_RARM_IDX].copy(), lo, hi
    )
    goal = _project_and_clamp(ctx, c.residual, goal_full[TORSO_RARM_IDX].copy(), lo, hi)
    return _plan(planner, start, goal, label, ARM_LINE_TIME)


def plan_body_free(planner, current_full, goal_full, label):
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    lp = make_leg_ratio(ctx, f"legratio_{_sanitize(label)}")
    _use(planner, BODY_SUBGROUP, current_full, [lp])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(ctx, lp.residual, current_full[BODY_IDX].copy(), lo, hi)
    goal = _project_and_clamp(ctx, lp.residual, goal_full[BODY_IDX].copy(), lo, hi)
    return _plan(planner, start, goal, label)


def plan_body_line(planner, current_full, goal_full, p_from, p_to, label):
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    leg_ratio = ctx.q[BODY_KNEE_IDX] - 2 * ctx.q[BODY_ANKLE_IDX]
    pf, pt = np.asarray(p_from, float), np.asarray(p_to, float)
    d = pt - pf
    d /= float(np.linalg.norm(d))
    seed = np.array([1, 0, 0.0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0.0])
    u = np.cross(d, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    gripper = ctx.link_translation(GRIPPER_LINK)
    diff = gripper - ca.DM(pf.tolist())
    line_res = ca.vertcat(
        ca.dot(diff, ca.DM(u.tolist())), ca.dot(diff, ca.DM(v.tolist()))
    )
    combined = ca.vertcat(leg_ratio, line_res)
    c = Constraint(residual=combined, q_sym=ctx.q, name=f"bodyline_{_sanitize(label)}")
    _use(planner, BODY_SUBGROUP, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(ctx, c.residual, current_full[BODY_IDX].copy(), lo, hi)
    goal = _project_and_clamp(ctx, c.residual, goal_full[BODY_IDX].copy(), lo, hi)
    return _plan(planner, start, goal, label, ARM_LINE_TIME)


def make_grip_rotation_lock(ctx, R_target, name, grip):
    """3-equation residual locking the gripper rotation to R_target (full SO(3))."""
    R = ctx.link_rotation(grip)
    R_t = ca.DM(np.asarray(R_target, float).tolist())
    M = R.T @ R_t - ca.DM_eye(3)
    res = ca.vertcat(M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1])
    return Constraint(residual=res, q_sym=ctx.q, name=name)


def plan_body_locked_free(planner, current_full, goal_full, R_target, label, grip):
    """Body-subgroup free-motion plan: leg ratio (``knee = 2·ankle``) +
    right-gripper rotation lock as hard manifold constraints, RRT-Connect
    with no cost.  This is a "hold the bowl exactly" motion — hard
    constraints are the right fit.
    """
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    R = ctx.link_rotation(grip)
    R_t = ca.DM(np.asarray(R_target, float).tolist())
    M = R.T @ R_t - ca.DM_eye(3)
    rot_res = ca.vertcat(M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1])
    leg_ratio = ctx.q[BODY_KNEE_IDX] - 2 * ctx.q[BODY_ANKLE_IDX]
    res = ca.vertcat(rot_res, leg_ratio)
    c = Constraint(residual=res, q_sym=ctx.q, name=f"bodylock_{_sanitize(label)}")
    _use(planner, BODY_SUBGROUP, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(ctx, c.residual, current_full[BODY_IDX].copy(), lo, hi)
    goal = _project_and_clamp(ctx, c.residual, goal_full[BODY_IDX].copy(), lo, hi)
    return _plan(planner, start, goal, label, ARM_LINE_TIME)


def plan_body_locked_line_direct(
    planner, current_full, goal_full, R_target, label, grip, steps=20
):
    """Construct a line motion path during holding by direct interpolation.

    Sampling-based planners struggle with the 7-eq constraint manifold (line
    + rotation lock + leg pin) on 21 DOF.  Since the line motion is a simple
    12 cm Cartesian translation between two manifold-connected configs, we
    construct the path by projecting interpolated targets directly.
    """
    ctx = SymbolicContext(BODY_SUBGROUP, base_config=current_full)
    gp = ctx.link_translation(grip)
    R = ctx.link_rotation(grip)
    R_t = ca.DM(np.asarray(R_target, float).tolist())
    M = R.T @ R_t - ca.DM_eye(3)
    rot_res = ca.vertcat(M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1])
    leg_ratio = ctx.q[BODY_KNEE_IDX] - 2 * ctx.q[BODY_ANKLE_IDX]

    planner.set_subgroup(BODY_SUBGROUP, base_config=current_full)
    lo = np.array(planner._planner.lower_bounds())
    hi = np.array(planner._planner.upper_bounds())

    start_pos = ctx.evaluate_link_pose(grip, current_full[BODY_IDX])[:3, 3]
    end_pos = ctx.evaluate_link_pose(grip, goal_full[BODY_IDX])[:3, 3]

    path_active = [current_full[BODY_IDX].copy()]
    q = current_full[BODY_IDX].copy()
    for s in range(1, steps + 1):
        alpha = s / steps
        target_pos = start_pos * (1 - alpha) + end_pos * alpha
        pos_res = gp - ca.DM(target_pos.tolist())
        res = ca.vertcat(pos_res, rot_res, leg_ratio)
        for _ in range(20):
            q = ctx.project(q, res, max_iters=200)
            q = np.clip(q, lo + 1e-5, hi - 1e-5)
        if not planner._planner.validate(q):
            raise RuntimeError(f"{label}: collision at step {s}/{steps}")
        path_active.append(q.copy())
    full_path = np.tile(current_full, (len(path_active), 1))
    full_path[:, BODY_IDX] = np.asarray(path_active)
    print(f"  [{label}] direct: {len(path_active)} wp")
    return full_path


def install_beam(planner, base_cloud):
    """Append a dense surface sampling of the low-hanging plank to the
    planner's pointcloud and return the beam points so the caller can
    reveal them in the viewer at the right moment (not at startup).
    """
    hx, hy, hz = BEAM_HALF_EXTENTS
    xs = np.linspace(-hx, hx, 80)
    ys = np.linspace(-hy, hy, 20)
    zs = np.linspace(-hz, hz, 6)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    bottom = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -hz)], axis=-1)
    top = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, hz)], axis=-1)
    # side faces (y = ±hy)
    gx2, gz2 = np.meshgrid(xs, zs, indexing="ij")
    side_ypos = np.stack([gx2.ravel(), np.full(gx2.size, hy), gz2.ravel()], axis=-1)
    side_yneg = np.stack([gx2.ravel(), np.full(gx2.size, -hy), gz2.ravel()], axis=-1)
    local = np.concatenate([bottom, top, side_ypos, side_yneg], axis=0)
    beam_points = (local + BEAM_CENTER[None, :]).astype(np.float32)

    augmented = np.concatenate(
        [np.asarray(base_cloud, dtype=np.float32), beam_points],
        axis=0,
    )
    planner.add_pointcloud(augmented)
    print(
        f"  beam installed: +{len(beam_points)} pts → planner cloud "
        f"{len(augmented)} pts (viewer reveal deferred)"
    )
    return beam_points


def plan_drive_with_squat(
    planner,
    current_full,
    goal_base,
    label,
    time_limit=20.0,
    cost_weight=200.0,
    planner_name="rrtc",
):
    """Plan base motion + squat in **one shot**: hard constraints on
    the structural relationships, soft cost for "stay tall / upper
    body straight".

    Active joints (6 DOF): base (x, y, yaw) and the height chain
    (ankle, knee, waist_pitch).  waist_yaw and the right arm stay
    frozen at their ``base_config`` values so the arm/torso
    kinematic chain is rigid in body frame.

    Hard constraints (ProjectedStateSpace manifold):

        * ``knee = 2 · ankle``  — leg ratio, the structural coupling
          between the leg joints.
        * ``yaw = current_yaw`` — base orientation fixed.
        * ``R_right_grip[0,2] = R_right_grip[1,2] = 0`` — horizontal
          hold on the right gripper: the gripper's local Z axis must
          stay aligned with world Z at every manifold sample, so the
          bowl stays level.  With the right arm + waist_yaw frozen,
          the only active joint that can tilt the gripper is
          ``waist_pitch``, so the projector auto-compensates it
          against ``ankle``.

    Soft cost (held throughout):

        * ``w · ankle²`` — "stay tall": keeps the robot at full
          height except where the beam forces it to dip.

    Default planner is RRT-Connect.  It ignores the cost but
    path-length simplification naturally trims the squat to the
    minimum the beam demands.  Optimal alternatives compatible with
    the projected state space: ``rrtstar``, ``prmstar``.
    """
    ctx = SymbolicContext(DRIVE_SQUAT_SUBGROUP, base_config=current_full)
    leg_ratio = ctx.q[DRIVE_SQUAT_KNEE_IDX] - 2 * ctx.q[DRIVE_SQUAT_ANKLE_IDX]
    yaw_lock = ctx.q[DRIVE_SQUAT_YAW_IDX] - ca.DM(float(current_full[2]))
    R_grip = ctx.link_rotation(RIGHT_GRIPPER_LINK)
    res = ca.vertcat(leg_ratio, yaw_lock, R_grip[0, 2], R_grip[1, 2])
    c = Constraint(residual=res, q_sym=ctx.q, name=f"drive_squat_{_sanitize(label)}")

    cost = Cost(
        expression=ctx.q[DRIVE_SQUAT_ANKLE_IDX] ** 2,
        q_sym=ctx.q,
        name=f"staytall_{_sanitize(label)}",
        weight=cost_weight,
    )

    _use(planner, DRIVE_SQUAT_SUBGROUP, current_full, [c])
    planner.set_costs([cost])
    lo, hi = _bounds(planner)

    start_active = current_full[DRIVE_SQUAT_IDX].copy()
    start = _project_and_clamp(ctx, c.residual, start_active, lo, hi)

    # Goal: arrive at the target base pose, legs back to standing
    # (ankle = knee = 0).  waist_pitch stays at its start value —
    # that's what the arm's grasp IK was solved against, so keeping
    # it preserves the bowl's world-frame orientation.
    goal_active = current_full[DRIVE_SQUAT_IDX].copy()
    goal_active[0:3] = np.asarray(goal_base, dtype=np.float64)
    goal_active[DRIVE_SQUAT_ANKLE_IDX] = 0.0
    goal_active[DRIVE_SQUAT_KNEE_IDX] = 0.0
    goal_active[DRIVE_SQUAT_WAIST_PITCH_IDX] = float(current_full[5])
    goal = _project_and_clamp(ctx, c.residual, goal_active, lo, hi)

    saved_name = planner._config.planner_name
    saved_simplify = planner._config.simplify
    planner._config.planner_name = planner_name
    # Keep simplification on for RRT-Connect (trims unnecessary squat
    # depth/duration); disable for asymptotically-optimal planners so
    # their cost-optimised path survives.
    planner._config.simplify = planner_name == "rrtc"
    try:
        path = _plan(planner, start, goal, label, time_limit)
    finally:
        planner._config.planner_name = saved_name
        planner._config.simplify = saved_simplify
        planner.clear_costs()

    end_err = float(np.linalg.norm(path[-1, DRIVE_SQUAT_IDX] - goal))
    if end_err > 0.05:
        raise RuntimeError(
            f"{label}: planner returned approximate solution "
            f"(end-to-goal distance {end_err:.3f})"
        )
    return path


def plan_arm_free_horizontal(
    planner,
    current_full,
    goal_full,
    label,
    subgroup=ARM_SUBGROUP,
    idx=TORSO_ARM_IDX,
    grip=GRIPPER_LINK,
):
    """Free arm plan with a horizontal-hold (level gripper) constraint."""
    ctx = SymbolicContext(subgroup, base_config=current_full)
    c = make_horizontal_hold(ctx, f"hhold_{_sanitize(label)}", grip)
    _use(planner, subgroup, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(ctx, c.residual, current_full[idx].copy(), lo, hi)
    goal = _project_and_clamp(ctx, c.residual, goal_full[idx].copy(), lo, hi)
    return _plan(planner, start, goal, label)


def plan_arm_line_horizontal(
    planner,
    current_full,
    goal_full,
    p_from,
    p_to,
    label,
    subgroup=ARM_SUBGROUP,
    idx=TORSO_ARM_IDX,
    grip=GRIPPER_LINK,
):
    """Line approach/retreat combined with horizontal-hold constraint (4 eq)."""
    ctx = SymbolicContext(subgroup, base_config=current_full)
    p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
    d = p_to - p_from
    d /= float(np.linalg.norm(d))
    seed = np.array([1, 0, 0.0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0.0])
    u = np.cross(d, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    gripper = ctx.link_translation(grip)
    diff = gripper - ca.DM(p_from.tolist())
    line_res = ca.vertcat(
        ca.dot(diff, ca.DM(u.tolist())), ca.dot(diff, ca.DM(v.tolist()))
    )
    R = ctx.link_rotation(grip)
    hhold_res = ca.vertcat(R[0, 2], R[1, 2])
    combined = ca.vertcat(line_res, hhold_res)
    c = Constraint(residual=combined, q_sym=ctx.q, name=f"linehhold_{_sanitize(label)}")
    _use(planner, subgroup, current_full, [c])
    lo, hi = _bounds(planner)
    start = _project_and_clamp(ctx, c.residual, current_full[idx].copy(), lo, hi)
    goal = _project_and_clamp(ctx, c.residual, goal_full[idx].copy(), lo, hi)
    return _plan(planner, start, goal, label, ARM_LINE_TIME)


def plan_base(planner, current_full, goal_base, label):
    _use(planner, BASE_SUBGROUP, current_full)
    return _plan(
        planner,
        current_full[:3].copy(),
        np.asarray(goal_base, dtype=np.float64),
        label,
        BASE_NAV_TIME,
    )


def plan_body_squat(planner, current_full, goal_full, label):
    _use(planner, BODY_SUBGROUP, current_full)
    return _plan(
        planner, current_full[BODY_IDX].copy(), goal_full[BODY_IDX].copy(), label
    )


# ── Playback ───────────────────────────────────────────────────────


@dataclass
class Segment:
    path: np.ndarray
    attach_body_id: int | None
    attach_local_tf: np.ndarray | None
    attach_link_idx: int | None
    banner: str
    reveal_points: np.ndarray | None = None  # drawn once when this segment starts


def find_link_index(env, name):
    for i in range(env.sim.client.getNumJoints(env.sim.skel_id)):
        if env.sim.client.getJointInfo(env.sim.skel_id, i)[12].decode("utf-8") == name:
            return i
    raise RuntimeError(f"link {name!r} not found")


def capture_local_transform(env, link_idx, body_id):
    c = env.sim.client
    lp, lq = (
        c.getLinkState(env.sim.skel_id, link_idx)[0],
        c.getLinkState(env.sim.skel_id, link_idx)[1],
    )
    lr = np.asarray(c.getMatrixFromQuaternion(lq)).reshape(3, 3)
    op, oq = c.getBasePositionAndOrientation(body_id)
    orr = np.asarray(c.getMatrixFromQuaternion(oq)).reshape(3, 3)
    tf = np.eye(4)
    tf[:3, :3] = lr.T @ orr
    tf[:3, 3] = lr.T @ (np.asarray(op) - np.asarray(lp))
    return tf


def apply_attachment(env, link_idx, body_id, local_tf):
    c = env.sim.client
    lp, lq = (
        c.getLinkState(env.sim.skel_id, link_idx)[0],
        c.getLinkState(env.sim.skel_id, link_idx)[1],
    )
    lr = np.asarray(c.getMatrixFromQuaternion(lq)).reshape(3, 3)
    wr = lr @ local_tf[:3, :3]
    wp = lr @ local_tf[:3, 3] + np.asarray(lp)
    m = wr
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0:
        s = float(np.sqrt(t + 1) * 2)
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2)
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2)
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = float(np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2)
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    c.resetBasePositionAndOrientation(body_id, wp.tolist(), [x, y, z, w])


def play_segments(env, segments, fps=60.0):
    c = env.sim.client
    dt = 1.0 / fps
    frames = [
        (si, ri) for si, seg in enumerate(segments) for ri in range(seg.path.shape[0])
    ]
    idx, n, playing, last_si = 0, len(frames), False, -1
    print("\nControls: SPACE play/pause   ←/→ step   close window to exit\n")
    for s in segments:
        print(f"  → {s.banner} ({s.path.shape[0]} wp)")
    try:
        while c.isConnected():
            si, ri = frames[idx]
            seg = segments[si]
            env.set_configuration(seg.path[ri])
            if (
                seg.attach_body_id is not None
                and seg.attach_local_tf is not None
                and seg.attach_link_idx is not None
            ):
                apply_attachment(
                    env, seg.attach_link_idx, seg.attach_body_id, seg.attach_local_tf
                )
            if si != last_si:
                print(f"[{si}] {seg.banner}")
                if seg.reveal_points is not None:
                    env.add_pointcloud(seg.reveal_points, pointsize=3)
                last_si = si
            keys = c.getKeyboardEvents()
            if ord(" ") in keys and keys[ord(" ")] & pb.KEY_WAS_TRIGGERED:
                playing = not playing
            elif (
                not playing
                and pb.B3G_LEFT_ARROW in keys
                and keys[pb.B3G_LEFT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx - 1) % n
            elif (
                not playing
                and pb.B3G_RIGHT_ARROW in keys
                and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_WAS_TRIGGERED
            ):
                idx = (idx + 1) % n
            elif playing:
                idx = (idx + 1) % n
            time.sleep(dt)
    except pb.error:
        pass


# ── Main ───────────────────────────────────────────────────────────


def main(pcd_stride: int = 1, visualize: bool = True) -> None:
    env = PyBulletEnv(autolife_robot_config, visualize=visualize)
    print("── scene setup ──")
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud)} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::2], pointsize=2)
    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=-90.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[-0.5, -0.3, 0.7],
        )

    # Create the planner ONCE — reused for every plan() call.
    import time as _time

    t_build = _time.perf_counter()
    planner = create_planner(
        "autolife",
        config=PlannerConfig(planner_name="rrtc"),
        pointcloud=cloud,
    )
    t_build = _time.perf_counter() - t_build
    print(f"  planner built in {t_build*1000:.0f} ms (reused for all calls)")

    current = HOME_JOINTS.copy()
    current[:3] = BASE_SQUAT_EAST
    # Zero waist yaw so the upper body is perfectly straight/vertical.
    current[6] = 0.0
    env.set_configuration(current)

    apple_id = place_graspable(env, "apple", APPLE_MESH_INIT)
    bowl_id = place_graspable(env, "bowl", BOWL_MESH_INIT)
    bottle_id = place_graspable(env, "bottle", BOTTLE_MESH_INIT)
    gripper_link = find_link_index(env, GRIPPER_LINK)
    client = env.sim.client
    segs: list[Segment] = []

    def add(
        path,
        label,
        attach_id=None,
        attach_tf=None,
        attach_link=None,
        reveal_points=None,
    ):
        segs.append(
            Segment(
                path=path,
                attach_body_id=attach_id,
                attach_local_tf=attach_tf,
                attach_link_idx=attach_link,
                banner=label,
                reveal_points=reveal_points,
            )
        )

    # ── Hold home pose so viewer sees the robot standing straight ──
    add(np.tile(current, (60, 1)), "home pose")

    # ── Stage 1a: squat with back-fold legs (body + waist_pitch pin) ──
    print("\n── stage 1a: squat (body subgroup + waist_pitch constraint) ──")
    squat_goal = current.copy()
    squat_goal[3:6] = SQUAT_STANCE
    path = plan_body_squat(planner, current, squat_goal, "s1a squat")
    add(path, "s1a squat")
    current = path[-1]

    # ── Stage 1b: pick apple under table (body + leg-pin) ──
    print("\n── stage 1b: pick apple under table (body + leg-pin constraint) ──")
    gp, pp = fk_line_endpoints(
        APPLE_GRASP_FULL, APPLE_PREGRASP_FULL, BODY_SUBGROUP, BODY_IDX
    )

    path = plan_body_free(planner, current, APPLE_PREGRASP_FULL, "s1b free→pregrasp")
    add(path, "s1b approach")
    current = path[-1]

    path = plan_body_line(planner, current, APPLE_GRASP_FULL, pp, gp, "s1b approach")
    add(path, "s1b approach line")
    current = path[-1]

    # Capture attachment
    env.set_configuration(APPLE_GRASP_FULL)
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    apple_tf = capture_local_transform(env, gripper_link, apple_id)

    path = plan_body_line(planner, current, APPLE_PREGRASP_FULL, gp, pp, "s1b lift")
    add(path, "s1b lift", apple_id, apple_tf, gripper_link)
    current = path[-1]

    # ── Stage 1c: stand up (body + waist_pitch pin, apple attached) ──
    print("\n── stage 1c: stand up (body subgroup + waist_pitch constraint) ──")
    stand_goal = current.copy()
    stand_goal[3:6] = STANDING_STANCE
    path = plan_body_squat(planner, current, stand_goal, "s1c stand")
    add(path, "s1c stand", apple_id, apple_tf, gripper_link)
    current = path[-1]

    # ── Nav closer to table for placement ──
    print("\n── nav → table placement position ──")
    path = plan_base(planner, current, BASE_PLACE_EAST, "nav→place pos")
    add(path, "nav→place pos", apple_id, apple_tf, gripper_link)
    current = path[-1]

    # ── Stage 2: place apple on table (torso+arm) ──
    print("\n── stage 2: place apple on table ──")
    gp2, pp2 = fk_line_endpoints(
        APPLE_PLACE_FULL, APPLE_PREPLACE_FULL, ARM_SUBGROUP, TORSO_ARM_IDX
    )

    path = plan_arm_free(planner, current, APPLE_PREPLACE_FULL, "s2 free→preplace")
    add(path, "s2 carry", apple_id, apple_tf, gripper_link)
    current = path[-1]

    path = plan_arm_line(planner, current, APPLE_PLACE_FULL, pp2, gp2, "s2 lower")
    add(path, "s2 lower", apple_id, apple_tf, gripper_link)
    current = path[-1]

    path = plan_arm_line(planner, current, APPLE_PREPLACE_FULL, gp2, pp2, "s2 retreat")
    add(path, "s2 retreat")
    current = path[-1]

    # ── Retract arm to home before navigating to kitchen ──
    print("\n── retract arm to home ──")
    retract_goal = current.copy()
    retract_goal[TORSO_ARM_IDX] = HOME_JOINTS[TORSO_ARM_IDX]
    retract_goal[6] = 0.0  # zero waist yaw
    path = plan_arm_free(planner, current, retract_goal, "retract arm→home")
    add(path, "retract arm")
    current = path[-1]

    # ── Stage 3: nav to kitchen counter ──
    print("\n── stage 3: nav → kitchen counter ──")
    path = plan_base(planner, current, BASE_NEAR_KITCHEN, "s3 nav→kitchen")
    add(path, "s3 nav→kitchen")
    current = path[-1]

    # ── Stage 4: pick bowl from kitchen counter (right arm) ──
    # Approach is unconstrained (no rotation lock yet — bowl not held).
    # Lift activates the holding constraints: full SO(3) gripper rotation
    # lock + leg pin (legs locked at standing) + left arm free (body
    # subgroup makes both arms active in the plan).
    print("\n── stage 4: pick bowl from kitchen counter (right arm) ──")
    r_gripper_link = find_link_index(env, RIGHT_GRIPPER_LINK)
    gp3, pp3 = fk_line_endpoints(
        BOWL_GRASP_FULL,
        BOWL_PREGRASP_FULL,
        RARM_SUBGROUP,
        TORSO_RARM_IDX,
        grip=RIGHT_GRIPPER_LINK,
    )

    path = plan_rarm_free(planner, current, BOWL_PREGRASP_FULL, "s4 free→pregrasp")
    add(path, "s4 approach")
    current = path[-1]

    path = plan_rarm_line(planner, current, BOWL_GRASP_FULL, pp3, gp3, "s4 approach")
    add(path, "s4 approach line")
    current = path[-1]

    # Capture bowl attachment (right gripper)
    env.set_configuration(BOWL_GRASP_FULL)
    client.resetBasePositionAndOrientation(
        bowl_id, BOWL_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    bowl_tf = capture_local_transform(env, r_gripper_link, bowl_id)

    # Lift with rotation lock + leg pin (left arm free)
    path = plan_body_locked_line_direct(
        planner,
        current,
        BOWL_PREGRASP_FULL,
        BOWL_HOLD_ROT,
        "s4 lift",
        grip=RIGHT_GRIPPER_LINK,
    )
    add(path, "s4 lift", bowl_id, bowl_tf, r_gripper_link)
    current = path[-1]

    # ── Install the low-hanging beam (bowl is now in hand) ────────
    # Only exists in the second half of the demo — the robot must
    # duck under it while the bowl stays level and the legs remain
    # coupled.  A soft cost biases the height toward "stay tall".
    # The viewer reveals the beam pointcloud when the first
    # post-grasp segment starts playing — not at startup.
    print("\n── install beam obstacle over kitchen→workstation corridor ──")
    beam_points = install_beam(planner, cloud)

    # ── Stage 5: drive to coffee table, auto-squat under the beam ─
    # Single-shot plan over base + height chain (6 DOF) with the leg
    # coupling and yaw lock as hard constraints and "stay tall" as a
    # soft cost.  The optimiser itself decides where to dip the body.
    print("\n── stage 5: drive → coffee (auto-squat under beam) ──")
    path = plan_drive_with_squat(
        planner,
        current,
        BASE_NEAR_COFFEE,
        "s5 drive→coffee",
        time_limit=20.0,
    )
    add(
        path,
        "s5 drive→coffee (auto-squat)",
        bowl_id,
        bowl_tf,
        r_gripper_link,
        reveal_points=beam_points,
    )
    current = path[-1]

    # ── Stage 6: place bowl on coffee table (rotation locked, leg pin) ──
    print("\n── stage 6: place bowl on coffee table ──")
    gp4, pp4 = fk_line_endpoints(
        BOWL_PLACE_FULL,
        BOWL_PREPLACE_FULL,
        RARM_SUBGROUP,
        TORSO_RARM_IDX,
        grip=RIGHT_GRIPPER_LINK,
    )

    # Carry to preplace: free body motion with rotation lock + leg pin.
    path = plan_body_locked_free(
        planner,
        current,
        BOWL_PREPLACE_FULL,
        BOWL_HOLD_ROT,
        "s6 carry",
        grip=RIGHT_GRIPPER_LINK,
    )
    add(path, "s6 carry", bowl_id, bowl_tf, r_gripper_link)
    current = path[-1]

    # Lower to place: line motion via direct interpolation.
    path = plan_body_locked_line_direct(
        planner,
        current,
        BOWL_PLACE_FULL,
        BOWL_HOLD_ROT,
        "s6 lower",
        grip=RIGHT_GRIPPER_LINK,
    )
    add(path, "s6 lower", bowl_id, bowl_tf, r_gripper_link)
    current = path[-1]

    # Retreat: bowl released, rotation lock no longer needed — regular line.
    path = plan_rarm_line(planner, current, BOWL_PREPLACE_FULL, gp4, pp4, "s6 retreat")
    add(path, "s6 retreat")
    current = path[-1]

    # ── Retract right arm to home before navigating to sofa ──
    print("\n── retract arm to home ──")
    retract_goal = current.copy()
    retract_goal[TORSO_RARM_IDX] = HOME_JOINTS[TORSO_RARM_IDX]
    retract_goal[6] = 0.0  # zero waist yaw
    path = plan_rarm_free(planner, current, retract_goal, "retract arm")
    add(path, "retract arm")
    current = path[-1]

    # ── Stage 7: nav to sofa ──
    print("\n── stage 7: nav → sofa ──")
    path = plan_base(planner, current, BASE_NEAR_SOFA, "s7 nav→sofa")
    add(path, "s7 nav→sofa")
    current = path[-1]

    # ── Stage 8: pick bottle from sofa ──
    print("\n── stage 8: pick bottle from sofa ──")
    gp5, pp5 = fk_line_endpoints(
        BOTTLE_GRASP_FULL, BOTTLE_PREGRASP_FULL, ARM_SUBGROUP, TORSO_ARM_IDX
    )

    path = plan_arm_free(planner, current, BOTTLE_PREGRASP_FULL, "s8 free→pregrasp")
    add(path, "s8 approach")
    current = path[-1]

    path = plan_arm_line(planner, current, BOTTLE_GRASP_FULL, pp5, gp5, "s8 approach")
    add(path, "s8 approach line")
    current = path[-1]

    env.set_configuration(BOTTLE_GRASP_FULL)
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    bottle_tf = capture_local_transform(env, gripper_link, bottle_id)

    path = plan_arm_line(planner, current, BOTTLE_PREGRASP_FULL, gp5, pp5, "s8 lift")
    add(path, "s8 lift", bottle_id, bottle_tf, gripper_link)
    current = path[-1]

    # ── Stage 9: nav to coffee table ──
    print("\n── stage 9: nav → coffee table ──")
    path = plan_base(planner, current, BASE_NEAR_COFFEE, "s9 nav→coffee")
    add(path, "s9 nav→coffee", bottle_id, bottle_tf, gripper_link)
    current = path[-1]

    # ── Stage 10: place bottle on coffee table ──
    print("\n── stage 10: place bottle on coffee table ──")
    gp6, pp6 = fk_line_endpoints(
        BOTTLE_PLACE_FULL, BOTTLE_PREPLACE_FULL, ARM_SUBGROUP, TORSO_ARM_IDX
    )

    path = plan_arm_free(planner, current, BOTTLE_PREPLACE_FULL, "s10 free→preplace")
    add(path, "s10 carry", bottle_id, bottle_tf, gripper_link)
    current = path[-1]

    path = plan_arm_line(planner, current, BOTTLE_PLACE_FULL, pp6, gp6, "s10 lower")
    add(path, "s10 lower", bottle_id, bottle_tf, gripper_link)
    current = path[-1]

    path = plan_arm_line(
        planner, current, BOTTLE_PREPLACE_FULL, gp6, pp6, "s10 retreat"
    )
    add(path, "s10 retreat")
    current = path[-1]

    # Reset for playback
    env.set_configuration(segs[0].path[0])
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    client.resetBasePositionAndOrientation(
        bowl_id, BOWL_MESH_INIT.tolist(), [0, 0, 0, 1]
    )
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_MESH_INIT.tolist(), [0, 0, 0, 1]
    )

    total = sum(s.path.shape[0] for s in segs)
    print(f"\n── ready: {total} total frames across {len(segs)} segments ──")
    if not visualize:
        return
    play_segments(env, segs)


if __name__ == "__main__":
    Fire(main)
