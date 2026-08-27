"""cuRoboV2 constrained IK stress test with PyBullet visualization.

This follows the target sequence in ``constrained_vis.py``. cuRobo collision
checking is disabled. ``solve_constrained`` enforces the Autolife stability
rules on the returned IK configuration::

    Joint_Knee = 2 * Joint_Ankle
    -10 <= Joint_Waist_Pitch - Joint_Ankle <= 60 degrees
    -75 <= Joint_Waist_Yaw <= 75 degrees

Waist pitch remains an independent cuRobo IK degree of freedom.

Controls:
    n - solve the displayed target, then advance to the next target
    q - quit

Usage:
    python examples/ik/curobo_v2_vis.py
    python examples/ik/curobo_v2_vis.py --num-seeds 256 --return-seeds 16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from autolife_planning.autolife import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.kinematics import CuroboV2IKSolver, create_ik_solver
from autolife_planning.types import CuroboV2IKConfig, SE3Pose

GROUPS = JOINT_GROUPS
SEED = np.concatenate(
    [
        HOME_JOINTS[GROUPS["legs"]],
        HOME_JOINTS[GROUPS["waist"]],
        HOME_JOINTS[GROUPS["left_arm"]],
    ]
)

# Indices in the 21-DOF, base-free configuration used by PyBulletEnv.
# The whole-body-left chain is legs, waist, then left arm.
CHAIN_TO_BODY = np.arange(11)

# The IK chain ends at the wrist pivot. For visualization, use the fixed
# gripper child so the axes sit on the visible hand instead of behind it.
DISPLAY_LINK = "Link_Left_Gripper"


def get_ee_link_index(env: PyBulletEnv, link_name: str) -> int:
    client = env.sim.client
    for index in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, index)
        if info[12].decode("utf-8") == link_name:
            return index
    raise ValueError(f"PyBullet link '{link_name}' was not found")


def draw_frame(
    env: PyBulletEnv,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    length: float = 0.08,
    width: float = 3.0,
) -> list[int]:
    client = env.sim.client
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    line_ids = []
    for axis_index, color in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        axis = np.zeros(3)
        axis[axis_index] = length
        endpoint = position + rotation @ axis
        line_ids.append(
            client.addUserDebugLine(
                position.tolist(), endpoint.tolist(), color, lineWidth=width
            )
        )
    return line_ids


def draw_frame_at_link(
    env: PyBulletEnv,
    link_index: int,
    *,
    length: float = 0.08,
    width: float = 3.0,
) -> list[int]:
    position, rotation = get_link_pose(env, link_index)
    return draw_frame(
        env,
        position,
        rotation,
        length=length,
        width=width,
    )


def get_link_pose(
    env: PyBulletEnv,
    link_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world pose of a PyBullet URDF link frame."""
    client = env.sim.client
    state = client.getLinkState(
        env.sim.skel_id,
        link_index,
        computeForwardKinematics=True,
    )
    # Entries 4 and 5 are the URDF link frame pose. Entries 0 and 1 are the
    # inertial center-of-mass frame and may not match the IK frame.
    position = np.asarray(state[4], dtype=np.float64)
    rotation = np.asarray(client.getMatrixFromQuaternion(state[5])).reshape(3, 3)
    return position, rotation


def fixed_child_offset(
    env: PyBulletEnv,
    parent_index: int,
    child_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed child pose expressed in the parent link frame."""
    parent_position, parent_rotation = get_link_pose(env, parent_index)
    child_position, child_rotation = get_link_pose(env, child_index)
    return (
        parent_rotation.T @ (child_position - parent_position),
        parent_rotation.T @ child_rotation,
    )


def transform_fixed_child(
    parent_pose: SE3Pose,
    offset_position: np.ndarray,
    offset_rotation: np.ndarray,
) -> SE3Pose:
    """Transform a parent-frame target to its fixed child's target pose."""
    return SE3Pose(
        parent_pose.position + parent_pose.rotation @ offset_position,
        parent_pose.rotation @ offset_rotation,
    )


def chain_to_body(joint_positions: np.ndarray) -> np.ndarray:
    body = HOME_JOINTS[3:].copy()
    body[CHAIN_TO_BODY] = np.asarray(joint_positions, dtype=np.float64)
    return body


def apply_solution(env: PyBulletEnv, joint_positions: np.ndarray) -> None:
    env.set_joint_states(chain_to_body(joint_positions))


def wait_action(env: PyBulletEnv, message: str) -> str:
    """Wait for ``n`` or ``q`` and return the selected action."""
    client = env.sim.client
    text_id = client.addUserDebugText(
        message,
        [0, 0, 1.5],
        textColorRGB=[0, 0, 0],
        textSize=1.5,
    )
    print(message)
    action = "q"
    try:
        while client.isConnected():
            keys = client.getKeyboardEvents()
            if ord("n") in keys and keys[ord("n")] & pb.KEY_WAS_TRIGGERED:
                action = "n"
                break
            if ord("q") in keys and keys[ord("q")] & pb.KEY_WAS_TRIGGERED:
                break
            time.sleep(0.01)
    except pb.error:
        pass
    finally:
        try:
            client.removeUserDebugItem(text_id)
        except pb.error:
            pass
    return action


def clear_debug(env: PyBulletEnv, item_ids: list[int]) -> None:
    for item_id in item_ids:
        env.sim.client.removeUserDebugItem(item_id)


def rotation_x(degrees: float) -> np.ndarray:
    return Rotation.from_euler("x", degrees, degrees=True).as_matrix()


def rotation_y(degrees: float) -> np.ndarray:
    return Rotation.from_euler("y", degrees, degrees=True).as_matrix()


def rotation_z(degrees: float) -> np.ndarray:
    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


def build_targets(home_pose: SE3Pose) -> list[tuple[str, SE3Pose]]:
    position = home_pose.position
    rotation = home_pose.rotation
    return [
        ("Front reach (+30cm x)", SE3Pose(position + [0.30, 0.0, 0.0], rotation)),
        ("High reach (+25cm z)", SE3Pose(position + [0.05, 0.0, 0.25], rotation)),
        (
            "Low reach (-60cm z)",
            SE3Pose(position + [0.15, 0.0, -0.60], rotation_x(45) @ rotation),
        ),
        ("Side reach (+25cm y)", SE3Pose(position + [0.0, 0.25, 0.0], rotation)),
        (
            "Cross-body (-20cm y)",
            SE3Pose(position + [0.15, -0.20, 0.0], rotation),
        ),
        (
            "Front-low floor (+25x, -55z)",
            SE3Pose(position + [0.25, 0.0, -0.55], rotation_x(60) @ rotation),
        ),
        (
            "High front (+20x, +30z)",
            SE3Pose(position + [0.20, 0.0, 0.30], rotation_y(-20) @ rotation),
        ),
        (
            "Wrist rotation (45deg Z + 30deg X)",
            SE3Pose(
                position + [0.10, 0.0, 0.0],
                rotation_z(45) @ rotation_x(30) @ rotation,
            ),
        ),
        ("Far front (+40cm x)", SE3Pose(position + [0.40, 0.05, 0.0], rotation)),
        (
            "Low side (+15y, -50z)",
            SE3Pose(position + [0.10, 0.15, -0.50], rotation_x(40) @ rotation),
        ),
    ]


def stability_metrics(
    solver: CuroboV2IKSolver,
    joint_positions: np.ndarray,
) -> tuple[float, float]:
    """Return leg-coupling residual and signed waist-minus-ankle angle."""
    names = solver.joint_names
    ankle = float(joint_positions[names.index("Joint_Ankle")])
    knee = float(joint_positions[names.index("Joint_Knee")])
    waist = float(joint_positions[names.index("Joint_Waist_Pitch")])
    return abs(knee - 2.0 * ankle), waist - ankle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=256)
    parser.add_argument("--return-seeds", type=int, default=16)
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be >= 1")
    if not 1 <= args.return_seeds <= args.num_seeds:
        parser.error("--return-seeds must be in [1, --num-seeds]")
    print("cuRoboV2 Constrained IK Stress Test - PyBullet Visualization")
    print("Collision checking: disabled")
    print("Frames: large = target, small = current/achieved gripper")
    print("=" * 70)

    env = PyBulletEnv(autolife_robot_config, visualize=True)
    ee_link = CHAIN_CONFIGS["whole_body_left"].ee_link
    ee_index = get_ee_link_index(env, ee_link)
    display_index = get_ee_link_index(env, DISPLAY_LINK)
    ee_to_display_position, ee_to_display_rotation = fixed_child_offset(
        env,
        ee_index,
        display_index,
    )

    config = CuroboV2IKConfig(
        num_seeds=args.num_seeds,
        return_seeds=args.return_seeds,
    )
    solver = create_ik_solver(
        "whole_body",
        side="left",
        backend="curobo_v2",
        config=config,
    )
    assert isinstance(solver, CuroboV2IKSolver)

    targets = build_targets(solver.fk(SEED))
    results: list[tuple[str, str, float, float, float, float]] = []
    total = len(targets)

    for index, (name, target) in enumerate(targets, start=1):
        display_target = transform_fixed_child(
            target,
            ee_to_display_position,
            ee_to_display_rotation,
        )
        print(f"\n[{index}/{total}] {name}")
        print(f"  wrist target: {np.round(target.position, 4)}")
        print(f"  gripper target: {np.round(display_target.position, 4)}")

        env.set_joint_states(HOME_JOINTS[3:])
        home_frame_items = draw_frame_at_link(
            env,
            display_index,
            length=0.06,
            width=2,
        )
        target_frame_items = draw_frame(
            env,
            display_target.position,
            display_target.rotation,
            length=0.10,
            width=4,
        )
        debug_items = home_frame_items + target_frame_items

        prompt = f"[{index}/{total}] {name}. Press 'n' to solve, 'q' to quit."
        if wait_action(env, prompt) == "q":
            clear_debug(env, debug_items)
            break

        start = time.perf_counter()
        result = solver.solve_constrained(target, seed=SEED)
        solve_ms = 1000.0 * (time.perf_counter() - start)
        print(f"  status: {result.status.value} ({solve_ms:.2f} ms)")
        print(f"  position error: {result.position_error:.6f} m")
        print(f"  orientation error: {result.orientation_error:.6f} rad")
        if not result.success and result.joint_positions is not None:
            print("  NOTE: failed candidate shown; target and achieved frames may differ")

        # The small frame drawn before solving belongs to the home pose. Once
        # the robot moves it becomes stale and looks like an FK/joint-mapping
        # error, so remove it before displaying the result. Keep the large
        # target frame for direct comparison with the achieved frame.
        clear_debug(env, home_frame_items)
        debug_items = target_frame_items

        leg_residual = float("nan")
        waist_difference = float("nan")
        if result.joint_positions is not None:
            leg_residual, waist_difference = stability_metrics(
                solver, result.joint_positions
            )
            print(f"  leg residual: {leg_residual:.3e} rad")
            print(f"  waist-ankle: {np.degrees(waist_difference):.2f} deg")
            apply_solution(env, result.joint_positions)
            debug_items += draw_frame_at_link(
                env,
                display_index,
                length=0.06,
                width=2,
            )
            achieved = solver.fk(result.joint_positions)
            achieved_display = transform_fixed_child(
                achieved,
                ee_to_display_position,
                ee_to_display_rotation,
            )
            print(f"  achieved wrist: {np.round(achieved.position, 4)}")
            print(f"  achieved gripper: {np.round(achieved_display.position, 4)}")

        results.append(
            (
                name,
                result.status.value,
                result.position_error,
                result.orientation_error,
                solve_ms,
                waist_difference,
            )
        )

        action = wait_action(
            env,
            f"[{index}/{total}] Done. Press 'n' for next, 'q' to quit.",
        )
        clear_debug(env, debug_items)
        if action == "q":
            break

    print("\n" + "=" * 100)
    print(
        "SUMMARY (cuRoboV2, knee coupled, -10 <= waist-ankle <= 60 degrees, "
        "-75 <= waist-yaw <= 75 degrees)"
    )
    print("=" * 100)
    print(
        f"{'Target':<35} {'Status':<10} {'Pos(mm)':>9} {'Ori(deg)':>10} "
        f"{'IK(ms)':>9} {'Waist(deg)':>11}"
    )
    print("-" * 100)
    success_count = 0
    for name, status, position_error, orientation_error, solve_ms, waist in results:
        marker = "" if status == "success" else " <--"
        waist_degrees = np.degrees(waist)
        print(
            f"{name:<35} {status:<10} {position_error * 1000.0:>9.2f} "
            f"{np.degrees(orientation_error):>10.2f} {solve_ms:>9.2f} "
            f"{waist_degrees:>11.2f}{marker}"
        )
        success_count += status == "success"
    print("-" * 100)
    print(f"Success: {success_count}/{len(results)}")

    if env.sim.client.isConnected():
        wait_action(env, "All targets done. Press 'q' to quit.")
    print("Done.")


if __name__ == "__main__":
    main()
