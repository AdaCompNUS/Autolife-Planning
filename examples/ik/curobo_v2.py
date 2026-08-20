"""cuRoboV2 whole-body IK correctness and speed test.

The constrained mode (default) hard-enforces the Autolife leg relation::

    Joint_Knee = 2 * Joint_Ankle

Waist pitch stays independent, with
``-10 <= Joint_Waist_Pitch - Joint_Ankle <= 60 degrees``.
Waist yaw is limited to ``[-75, 75] degrees``.

cuRobo collision checking is disabled by this backend.  Both timed calls use
the true batched GPU API.  The cold batch includes lazy CUDA/cuRobo
construction and kernel warm-up; the second batch measures the warm solver.

Usage:
    python examples/ik/curobo_v2.py
    python examples/ik/curobo_v2.py --queries 100 --num-seeds 256
    python examples/ik/curobo_v2.py --queries 1000 --batch-size 100
    python examples/ik/curobo_v2.py --unconstrained
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from autolife_planning.autolife import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.types import CuroboV2IKConfig


def _whole_body_home(side: str) -> np.ndarray:
    groups = JOINT_GROUPS
    arm = "left_arm" if side == "left" else "right_arm"
    return np.concatenate(
        [
            HOME_JOINTS[groups["legs"]],
            HOME_JOINTS[groups["waist"]],
            HOME_JOINTS[groups[arm]],
        ]
    )


def _reachable_targets(solver, home: np.ndarray, count: int, seed: int):
    """Generate guaranteed-reachable poses from nearby stable joint states."""
    rng = np.random.default_rng(seed)
    targets = []
    for _ in range(count):
        q = home.copy()
        ankle = rng.uniform(0.0, 0.12)
        q[0] = ankle
        q[1] = 2.0 * ankle
        q[2] = ankle + rng.uniform(np.radians(-9.0), np.radians(9.0))
        q[3] += rng.uniform(-0.04, 0.04)
        q[4:] += rng.uniform(-0.04, 0.04, size=q.shape[0] - 4)
        targets.append(solver.fk(q))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "fixed cuRobo GPU batch/chunk size; defaults to --queries so the "
            "benchmark uses one solve_pose call"
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=256)
    parser.add_argument("--return-seeds", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="test ordinary cuRobo IK instead of the stability-constrained solve",
    )
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("--queries must be >= 1")
    batch_size = args.queries if args.batch_size is None else args.batch_size
    if batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.return_seeds > args.num_seeds:
        parser.error("--return-seeds must be <= --num-seeds")

    config = CuroboV2IKConfig(
        num_seeds=args.num_seeds,
        return_seeds=args.return_seeds,
        random_seed=args.random_seed,
        max_batch_size=batch_size,
    )
    solver = create_ik_solver(
        "whole_body",
        side=args.side,
        backend="curobo_v2",
        config=config,
    )
    home = _whole_body_home(args.side)
    targets = _reachable_targets(solver, home, args.queries, args.random_seed)
    seeds = np.repeat(home[None, :], args.queries, axis=0)
    solve_batch = (
        solver.solve_batch
        if args.unconstrained
        else solver.solve_constrained_batch
    )
    mode = "ordinary" if args.unconstrained else "stability-constrained"

    print(f"Chain: {solver.base_frame} -> {solver.ee_frame}")
    print(f"Mode: {mode}; collision checking: disabled")
    print(f"Queries: {args.queries}; cuRobo seeds/query: {args.num_seeds}")
    print(f"GPU batch/chunk size: {batch_size}")

    start = time.perf_counter()
    cold_results = solve_batch(targets, seeds)
    cold_seconds = time.perf_counter() - start
    cold_successes = sum(result.success for result in cold_results)
    print(
        f"Cold {args.queries}-query GPU batch (includes lazy solver build): "
        f"{cold_seconds:.3f} s [{cold_successes}/{args.queries} success]"
    )

    batch_start = time.perf_counter()
    results = solve_batch(targets, seeds)
    batch_seconds = time.perf_counter() - batch_start
    successes = [result for result in results if result.success]

    print(f"Warm {args.queries}-query batch: {batch_seconds:.3f} s")
    print(f"  throughput: {args.queries / batch_seconds:.1f} queries/s")
    print(
        f"  amortized latency: {1e3 * batch_seconds / args.queries:.2f} ms/query"
    )
    print(f"  success: {len(successes)}/{args.queries}")

    if successes:
        max_position_error = max(result.position_error for result in successes)
        max_orientation_error = max(result.orientation_error for result in successes)
        print(f"  max position error: {max_position_error:.6f} m")
        print(f"  max orientation error: {max_orientation_error:.6f} rad")

    if successes and not args.unconstrained:
        names = solver.joint_names
        ankle_i = names.index("Joint_Ankle")
        knee_i = names.index("Joint_Knee")
        waist_i = names.index("Joint_Waist_Pitch")
        leg_residual = max(
            abs(
                result.joint_positions[knee_i]
                - 2.0 * result.joint_positions[ankle_i]
            )
            for result in successes
        )
        waist_differences = [
            result.joint_positions[waist_i] - result.joint_positions[ankle_i]
            for result in successes
        ]
        print(f"  max leg-coupling residual: {leg_residual:.3e} rad")
        print(
            "  waist-ankle range: "
            f"[{np.degrees(min(waist_differences)):.2f}, "
            f"{np.degrees(max(waist_differences)):.2f}] deg"
        )


if __name__ == "__main__":
    main()
