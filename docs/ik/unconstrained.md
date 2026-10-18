# Unconstrained IK (TRAC-IK)

TRAC-IK wraps the KDL-derived [TRAC-IK C++ solver](https://bitbucket.org/traclabs/trac_ik).
It runs a sequential-quadratic-programming solver and a Newton
pseudoinverse solver **concurrently** and returns whichever finds a
valid solution first, with optional weighting of the solution by
distance, manipulability, or raw speed.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/trac_ik.mp4" type="video/mp4">
</video>

## Minimal example

```python
import numpy as np
from autolife_planning.autolife import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

solver = create_ik_solver(
    "left_arm",
    config=IKConfig(
        timeout=0.2,          # seconds per attempt
        epsilon=1e-5,          # convergence tolerance
        solve_type=SolveType.SPEED,
        max_attempts=10,
        position_tolerance=1e-4,
        orientation_tolerance=1e-4,
    ),
)

home = HOME_JOINTS[JOINT_GROUPS["left_arm"]]
ee = solver.fk(home)                    # forward kinematics

target = SE3Pose(
    position=ee.position + np.array([0.05, 0.0, -0.05]),
    rotation=ee.rotation,
)
result = solver.solve(target, seed=home)
print(result.status, result.joint_positions)
```

## Solve types

`IKConfig.solve_type` picks the post-solution objective when multiple
valid solutions exist in the timeout window:

| `SolveType` | Picks |
|---|---|
| `SPEED` | whichever solver finishes first |
| `DISTANCE` | solution closest to the seed |
| `MANIP1` | solution with highest manipulability (Jacobian SVD product) |
| `MANIP2` | solution with the highest ratio of smallest to largest Jacobian singular value |

## When to use

- **Pick-and-place, reach-to-grasp, visual servoing** — one
  end-effector pose, no extra task-space invariants.
- **Fast enough for online use** — default 0.2 s timeout is
  conservative; most calls return in a few milliseconds.
- **Seeded batches** — pass a seed close to the current joint state
  to get smooth solution continuity across frames.

## API reference

See [Kinematics → TracIKSolver](../api/kinematics.md#autolife_planning.kinematics.trac_ik_solver.TracIKSolver).
