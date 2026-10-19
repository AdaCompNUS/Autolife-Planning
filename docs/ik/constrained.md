# Constrained IK (Pink)

Pink solves IK as a quadratic program at every time step. A primary
task — usually the end-effector pose — is composed with soft
secondary objectives: centre-of-mass stability, camera-frame
stabilization, self-collision avoidance, and a posture regularizer.
Each step solves for the joint velocity that best satisfies all of
them under the given weights and integrates it; `solve()` returns the
final joint positions and `solve_constrained()` the whole sequence.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_ik.mp4" type="video/mp4">
</video>

## Minimal example

```python
import numpy as np
from autolife_planning.autolife import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.types import PinkIKConfig, SE3Pose

solver = create_ik_solver(
    "whole_body",
    side="left",
    backend="pink",
    config=PinkIKConfig(
        dt=0.01,
        max_iterations=300,
        position_cost=1.0,
        orientation_cost=1.0,
        com_cost=0.1,              # hold the CoM where it is at the seed
        lm_damping=1e-3,           # Levenberg-Marquardt damping
        posture_cost=1e-3,         # soft pull toward seed posture
        self_collision=True,       # add the collision barrier ...
    ),
    self_collision=True,            # ... over the SRDF's collision pairs
)

G = JOINT_GROUPS
home = np.concatenate([HOME_JOINTS[G["legs"]], HOME_JOINTS[G["waist"]], HOME_JOINTS[G["left_arm"]]])
ee = solver.fk(home)
target = SE3Pose(
    position=ee.position + np.array([0.15, -0.05, 0.20]),
    rotation=ee.rotation,
)

result = solver.solve(target, seed=home)
print(result.status, result.joint_positions)
```

## Secondary objectives

Every objective is a cost term on the step's QP objective; `*_cost`
fields tune the weight relative to the primary end-effector task.

| Field | Effect |
|---|---|
| `position_cost` / `orientation_cost` | weight of the primary EE task |
| `com_cost` | holds the CoM at its seed position — set non-zero for whole-body chains |
| `camera_frame` + `camera_cost` | holds the camera frame's full pose at its seed pose |
| `posture_cost` | pulls unused joints toward the seed posture |
| `lm_damping` | Levenberg-Marquardt damping near singularities |
| `self_collision` | adds collision-pair barriers from the SRDF; also pass `self_collision=True` to `create_ik_solver` so the collision model is built |

## When to use

- **Whole-body reaching** with the mobile base included — TRAC-IK
  gives you a solution, Pink keeps the CoM over the support polygon
  while you reach for it.
- **Camera-stable head tracking** — pin the camera-frame task while
  the arm moves.
- **Smooth reaching motions** — `solve_constrained()` also returns
  the configuration after every QP step of `dt`.

## API reference

See [Kinematics → PinkIKSolver](../api/kinematics.md#autolife_planning.kinematics.pink_ik_solver.PinkIKSolver).
