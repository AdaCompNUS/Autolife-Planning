# Constrained IK (Pink)

Pink solves IK as a quadratic program at every time step. A primary
task — usually the end-effector pose — is composed with any number
of soft secondary objectives: centre-of-mass stability, camera-frame
stabilization, self-collision avoidance, joint coupling, and a
posture regularizer. Each step returns a joint-velocity command
that best satisfies all of them under the given weights.

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
        com_cost=0.1,              # keep CoM above the support polygon
        lm_damping=1e-3,           # Levenberg-Marquardt damping
        posture_cost=1e-3,         # soft pull toward seed posture
    ),
    self_collision=True,            # parse URDF's SRDF into pairs
)

home = HOME_JOINTS[JOINT_GROUPS["whole_body_left"]]
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
| `com_cost` | pulls the projected CoM over the support polygon — set non-zero for whole-body chains that include the base |
| `camera_frame` + `camera_cost` | pins a head/camera link's yaw-pitch toward a target |
| `posture_cost` | pulls unused joints toward the seed posture |
| `lm_damping` | Levenberg-Marquardt damping near singularities |
| `coupled_joints` | master/slave linear couplings |
| `self_collision` | adds collision-pair barriers from the URDF's SRDF |

## When to use

- **Whole-body reaching** with the mobile base included — TRAC-IK
  gives you a solution, Pink keeps the CoM over the support polygon
  while you reach for it.
- **Camera-stable head tracking** — pin the camera-frame task while
  the arm moves.
- **Dual-arm coupling** — express "the two grippers stay 30 cm
  apart" as a coupled-joint constraint on top of the primary task.
- **Continuous control loops** — Pink returns per-step velocities,
  so embed it inside a 50–100 Hz controller.

## API reference

See [Kinematics → PinkIKSolver](../api/kinematics.md#autolife_planning.kinematics.pink_ik_solver.PinkIKSolver).
