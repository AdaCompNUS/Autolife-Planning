"""Time parameterization: convert a planned path into an executable trajectory.

Demonstrates the default TOPP-RA time-optimal parameterizer. A synthetic
3-DOF zigzag path is parameterized under velocity and acceleration limits,
then sampled at 100 Hz and printed.

    pixi run python examples/planning/time_parameterization.py
"""

import numpy as np
from fire import Fire

from autolife_planning.trajectory import TimeOptimalParameterizer


def main(
    vel_limit: float = 1.0,
    acc_limit: float = 2.0,
    dt: float = 0.01,
    method: str = "toppra",
) -> None:
    """Run time parameterization on a synthetic path.

    Args:
        vel_limit: Per-joint velocity limit (rad/s or m/s).
        acc_limit: Per-joint acceleration limit (rad/s^2 or m/s^2).
        dt: Sample interval for the output rollout (seconds).
        method: Time-parameterization backend: ``toppra`` or ``totg``.
    """
    ndof = 3
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.3, -0.2],
            [1.0, 0.6, 0.1],
            [1.2, 0.4, 0.3],
        ]
    )

    vel_limits = np.full(ndof, vel_limit)
    acc_limits = np.full(ndof, acc_limit)

    param = TimeOptimalParameterizer(
        vel_limits,
        acc_limits,
        max_deviation=0.1,
        method=method,
    )
    traj = param.parameterize(path)

    print(f"Path: {path.shape[0]} waypoints, {ndof} DOF")
    print(f"Vel limits: {vel_limits}")
    print(f"Acc limits: {acc_limits}")
    print(f"Duration:   {traj.duration:.4f} s")
    print()

    times, positions, velocities, accelerations = traj.sample_uniform(dt)

    print(f"Sampled at dt={dt}s -> {len(times)} points")
    print(f"  Start position:  {positions[0]}")
    print(f"  End position:    {positions[-1]}")
    print(f"  Start velocity:  {velocities[0]}")
    print(f"  End velocity:    {velocities[-1]}")
    print(f"  Max |velocity|:  {np.abs(velocities).max(axis=0)}")
    print(f"  Max |accel|:     {np.abs(accelerations).max(axis=0)}")

    # Verify bounds.
    vel_ok = np.all(np.abs(velocities) <= vel_limits + 1e-6)
    print(f"\n  Velocity within limits: {vel_ok}")


if __name__ == "__main__":
    Fire(main)
