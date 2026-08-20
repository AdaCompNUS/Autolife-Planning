# Autolife Planning

A planning library for the **Autolife robot** — inverse kinematics
(TRAC-IK and Pink QP), motion planning (OMPL frontend, VAMP SIMD
collision backend), and time-optimal trajectory generation, all behind
a unified Python API.

<video controls autoplay loop muted playsinline width="100%">
  <source src="assets/rls_pick_place.mp4" type="video/mp4">
</video>

<div class="grid cards" markdown>

-   :material-package-variant:{ .lg .middle } __Drop-in, few deps__

    ---

    For inference: **`numpy`, `scipy`, `pink`**. No conda required, no
    ROS, no MoveIt. Vendor it into any project with a single
    `pip install -e .`.

-   :material-source-branch:{ .lg .middle } __26 planners__

    ---

    Full OMPL planner library, not a bespoke subset: RRT-Connect,
    RRT\*, BIT\*, AIT\*, EIT\*, KPIECE, PRM, FMT, SPARS, … plug in
    whichever suits the problem.

-   :material-speedometer:{ .lg .middle } __Microsecond checks, millisecond plans__

    ---

    VAMP SIMD collision backend: **~3 μs per check**. RRT-Connect plans
    a **1.1 ms** collision-free path for the 7-, 14-, and 24-DOF body.

</div>

## Features

<div class="grid cards" markdown>

-   :material-robot-angry-outline:{ .lg .middle } [__Inverse Kinematics__](ik/index.md)

    ---

    TRAC-IK for unconstrained numerical IK; Pink for QP-based
    constrained IK that composes end-effector tracking with
    centre-of-mass stability, camera-frame stabilization, and
    self-collision avoidance.

-   :material-map-marker-path:{ .lg .middle } [__Motion Planning__](planning/index.md)

    ---

    OMPL frontend + VAMP SIMD backend. Full-body and subgroup planning
    (single-arm, dual-arm, torso + arm, base, whole-body) with
    first-class support for point-cloud obstacles.

-   :material-vector-curve:{ .lg .middle } [__Manifold Planning__](planning/manifold.md)

    ---

    Hard task-space equality constraints — planes, rails, orientation
    locks, couplings — compiled from CasADi expressions and solved on
    OMPL's `ProjectedStateSpace`.

-   :material-chart-bell-curve:{ .lg .middle } [__Cost-space Planning__](planning/cost.md)

    ---

    Path-integral soft costs (orientation preference, pose
    stabilization, …) compiled from CasADi and driving OMPL's
    asymptotically-optimal planners (RRT\*, BIT\*, AIT\*, …).

-   :material-timer-outline:{ .lg .middle } [__Time Parameterization__](planning/trajectory.md)

    ---

    TOPP-RA converts geometric paths into executable trajectories with
    per-joint velocity and acceleration limits — bridging the planner's
    output to hardware.

</div>

## Benchmarks

Median / p95 wall-clock for a single **search** (simplification and
interpolation disabled), N = 20 runs per cell, plans from the home
pose to a sampled collision-free goal around a tabletop point-cloud
obstacle (same scene as `examples/planning/motion.py`).

| Planner | Left arm (7 DOF) | Dual arm (14 DOF) | Full body (24 DOF) |
|---|---:|---:|---:|
| RRT-Connect | **1.1 / 1.1 ms** | **1.1 / 1.2 ms** | **1.1 / 1.1 ms** |
| KPIECE | 1.1 / 2.3 ms | 1.7 / 3.2 ms | 1.1 / 4.3 ms |
| PRM | 1.1 / 3.2 ms | 1.1 / 3.2 ms | 2.2 / 4.2 ms |
| SBL | 1.1 / 1.1 ms | 2.2 / 3.3 ms | 8.5 / 17.0 ms |
| EST / BiEST | 1.1 / 4.3 ms | 2.1 / 5.3 ms | 2.2 / 6.4 ms |
| STRIDE | 2.2 / 3.2 ms | 2.7 / 8.5 ms | 2.2 / 3.3 ms |

Raw `planner.validate(config)` throughput: **3.2 μs / check** (Python
boundary included) ≈ **0.31 M checks / s**. Batched
[`planner.validate_batch(configs)`](planning/index.md#standalone-collision-checking)
packs up to `rake` distinct configs per SIMD block for an **~8.5×**
best-case speedup on all-valid workloads.

!!! note "Environment"
    13th Gen Intel Core i9-13900KF, 24 logical cores · Linux 6.8 ·
    Python 3.12.13. Reproduce with
    `pixi run -e dev python scripts/benchmarks/motion_planning.py`.
    See [planning](planning/index.md) for the full planner sweep including
    asymptotically-optimal variants (RRT\*, BIT\*, AIT\*, EIT\*, FMT).

## Quick Install

For inference — running the planners and IK solvers — just pip install:

```bash
git clone --recursive https://github.com/AdaCompNUS/Autolife-Planning.git
cd Autolife-Planning
pip install -e .
```

Three runtime deps: `numpy`, `scipy`, `pink`. No conda, no ROS, no
MoveIt. See the [Getting Started](getting-started.md) guide for the
full development setup (pixi + conda-forge toolchain, URDF rebuilds,
FK codegen).
