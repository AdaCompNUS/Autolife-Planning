"""Solve one 20-query cuRoboV2 batch and inspect its IK results in PyBullet.

All batch entries use the same target and different stable whole-body seeds.
For 20 evenly spaced values of ``N`` from 0.0 through 0.8, the seeds use::

    Joint_Ankle = N
    Joint_Knee = 2 * N
    Joint_Waist_Pitch = N

The remaining waist and left-arm joints retain their home values. A single
robot displays one final batch result at a time; no trajectories are animated.

Controls:
    n - solve the batch, then advance to the next result
    q - quit

Usage:
    python examples/ik/curobo_v2_batch_vis.py
    python examples/ik/curobo_v2_batch_vis.py --target 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from autolife_planning.autolife import (
    CHAIN_CONFIGS,
    autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.kinematics import CuroboV2IKSolver, create_ik_solver
from autolife_planning.types import CuroboV2IKConfig

from curobo_v2_vis import (
    DISPLAY_LINK,
    SEED,
    apply_solution,
    build_targets,
    clear_debug,
    draw_frame,
    draw_frame_at_link,
    fixed_child_offset,
    get_ee_link_index,
    stability_metrics,
    transform_fixed_child,
    wait_action,
)

BATCH_SIZE = 20
MAX_N = 0.8


def build_stable_seeds() -> tuple[np.ndarray, np.ndarray]:
    """Return N values and a ``(20, 11)`` stable seed matrix."""
    values = np.linspace(0.0, MAX_N, BATCH_SIZE, dtype=np.float64)
    seeds = np.repeat(SEED[None, :], BATCH_SIZE, axis=0)
    seeds[:, 0] = values
    seeds[:, 1] = 2.0 * values
    seeds[:, 2] = values
    return values, seeds


def print_result(
    index: int,
    value: float,
    result,
    solver: CuroboV2IKSolver,
) -> None:
    print(f"\n[{index:02d}/{BATCH_SIZE}] seed N={value:.4f}")
    print(f"  status: {result.status.value}")
    print(f"  position error: {result.position_error:.6f} m")
    print(f"  orientation error: {result.orientation_error:.6f} rad")
    if result.joint_positions is None:
        print("  no IK candidate; displaying the input seed")
        return
    leg_residual, waist_difference = stability_metrics(
        solver,
        result.joint_positions,
    )
    print(f"  leg residual: {leg_residual:.3e} rad")
    print(f"  |waist-ankle|: {np.degrees(waist_difference):.2f} deg")
    if not result.success:
        print("  NOTE: candidate did not meet the requested pose tolerances")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=int,
        choices=range(1, 11),
        default=1,
        metavar="1..10",
        help="target from curobo_v2_vis.py (default: 1)",
    )
    parser.add_argument("--num-seeds", type=int, default=256)
    parser.add_argument("--return-seeds", type=int, default=16)
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be >= 1")
    if not 1 <= args.return_seeds <= args.num_seeds:
        parser.error("--return-seeds must be in [1, --num-seeds]")

    # max_batch_size=20 makes the 20 repeated targets execute in one cuRobo
    # solve_pose call instead of being split into smaller chunks.
    config = CuroboV2IKConfig(
        num_seeds=args.num_seeds,
        return_seeds=args.return_seeds,
        max_batch_size=BATCH_SIZE,
    )
    solver = create_ik_solver(
        "whole_body",
        side="left",
        backend="curobo_v2",
        config=config,
    )
    assert isinstance(solver, CuroboV2IKSolver)

    target_name, wrist_target = build_targets(solver.fk(SEED))[args.target - 1]
    values, seeds = build_stable_seeds()

    print("cuRoboV2 20-Query Batch IK Visualization")
    print(f"Target {args.target}: {target_name}")
    print("One GPU batch: 20 repeated targets with 20 different seeds")
    print("Seeds: (ankle, knee, waist pitch) = (N, 2N, N), 0 <= N <= 0.8")
    print("Collision checking: disabled")
    print("=" * 78)

    env = PyBulletEnv(autolife_robot_config, visualize=True)
    ee_index = get_ee_link_index(env, CHAIN_CONFIGS["whole_body_left"].ee_link)
    display_index = get_ee_link_index(env, DISPLAY_LINK)
    ee_to_display_position, ee_to_display_rotation = fixed_child_offset(
        env,
        ee_index,
        display_index,
    )
    gripper_target = transform_fixed_child(
        wrist_target,
        ee_to_display_position,
        ee_to_display_rotation,
    )
    target_items = draw_frame(
        env,
        gripper_target.position,
        gripper_target.rotation,
        length=0.10,
        width=4,
    )

    if wait_action(
        env,
        f"{target_name}. Press 'n' to solve one batch of 20, 'q' to quit.",
    ) == "q":
        clear_debug(env, target_items)
        return

    start = time.perf_counter()
    results = solver.solve_constrained_batch(
        [wrist_target] * BATCH_SIZE,
        seeds,
    )
    batch_ms = 1000.0 * (time.perf_counter() - start)
    print(f"Batch solve: {batch_ms:.2f} ms ({batch_ms / BATCH_SIZE:.2f} ms/query)")
    print(f"Success: {sum(result.success for result in results)}/{BATCH_SIZE}")

    result_items: list[int] = []
    quit_requested = False
    for index, (value, seed, result) in enumerate(
        zip(values, seeds, results),
        start=1,
    ):
        clear_debug(env, result_items)
        result_items = []

        if result.joint_positions is None:
            apply_solution(env, seed)
        else:
            apply_solution(env, result.joint_positions)
            result_items += draw_frame_at_link(
                env,
                display_index,
                length=0.06,
                width=2,
            )

        print_result(index, float(value), result, solver)
        status_color = [0.0, 0.55, 0.0] if result.success else [0.8, 0.0, 0.0]
        result_items.append(
            env.sim.client.addUserDebugText(
                f"Batch result {index}/20\nN={value:.4f}  {result.status.value}",
                [0.0, 0.0, 1.65],
                textColorRGB=status_color,
                textSize=1.2,
            )
        )

        if index < BATCH_SIZE:
            message = (
                f"[{index}/{BATCH_SIZE}] N={value:.4f}. "
                "Press 'n' for next result, 'q' to quit."
            )
        else:
            message = (
                f"[{index}/{BATCH_SIZE}] N={value:.4f}. "
                "Press 'n' for summary, 'q' to quit."
            )
        if wait_action(env, message) == "q":
            quit_requested = True
            break

    print("\n" + "=" * 78)
    print(f"Batch time: {batch_ms:.2f} ms")
    print(f"Success: {sum(result.success for result in results)}/{BATCH_SIZE}")
    print("Large axes are the target; small axes are the displayed batch result.")
    if not quit_requested and env.sim.client.isConnected():
        wait_action(env, "Batch complete. Press 'q' to quit.")


if __name__ == "__main__":
    main()
