# Time Parameterization

The missing step between a geometric path and a command stream that a
motor controller can execute. The motion planner gives you **where** to
go — a list of waypoints in joint space — but not **when**. Time
parameterization adds the timing: it assigns a time stamp to every
point on the path so that per-joint velocity and acceleration limits
are respected, and the trajectory is as fast as physically possible.

<div class="grid cards" markdown>

-   __Time-optimal__

    ---

    Finds the fastest feasible velocity profile along the path.
    Every joint is at its velocity or acceleration limit at every
    instant — there is no slack left to speed up.

-   __Bounded velocity + acceleration__

    ---

    Per-joint velocity and acceleration limits are hard constraints.
    The output trajectory never exceeds them (up to the integrator's
    numerical tolerance).

-   __Backend options__

    ---

    TOPP-RA is the default. The vendored MoveIt-style TOTG backend is
    still available for comparison or compatibility with older scripts.

</div>

## Algorithm

The default implementation is **TOPP-RA** (Time-Optimal Path
Parameterization by Reachability Analysis). It computes a feasible
velocity profile along a smooth geometric path while respecting
per-joint velocity and acceleration bounds.

The default flow works in two stages:

1. **Spline path through waypoints** — the waypoint path is represented
   as a natural cubic spline using chord-length parameterization. This
   avoids the hard waypoint corners that force stop-go timing profiles.
2. **Reachability analysis** — TOPP-RA finds the fastest feasible
   profile $\dot{s}(s)$ along that path under the joint limits.

The result is a trajectory $q(t)$ with continuous velocity and bounded
acceleration that starts and ends at rest.

!!! note "Collision fidelity"

    TOPP-RA only adds timing, but its smooth spline can deviate slightly
    between waypoints. Keep the planned waypoint path dense enough for
    your clearance, and collision-check the sampled trajectory when
    operating near obstacles.

!!! note "Legacy TOTG"

    Pass `method="totg"` to use the vendored Kunz-Stilman / MoveIt-style
    Time-Optimal Trajectory Generation backend. TOTG uses circular blends
    at corners, controlled by `max_deviation`.

## Minimal example

```python
import numpy as np
from autolife_planning.planning import create_planner
from autolife_planning.trajectory import TimeOptimalParameterizer
from autolife_planning.types import PlannerConfig

# 1. Plan a collision-free path. Keep interpolation/densification on when
#    you need the timed spline to stay close to the checked path.
planner = create_planner(
    "autolife_left_arm",
    config=PlannerConfig(simplify=True, interpolate=True),
)
start = planner.extract_config(home_joints)
goal  = planner.sample_valid()
result = planner.plan(start, goal)
path = result.path                       # (N, 7)

# 2. Time-parameterize.
vel_limits = np.full(planner.num_dof, 1.0)   # rad/s
acc_limits = np.full(planner.num_dof, 2.0)   # rad/s^2

param = TimeOptimalParameterizer(vel_limits, acc_limits)
traj  = param.parameterize(path)

print(f"Duration: {traj.duration:.3f} s")

# 3. Sample at controller rate.
times, positions, velocities, accelerations = traj.sample_uniform(dt=0.01)
```

## Configuration knobs

| Parameter | Default | Description |
|---|---|---|
| `max_velocity` | *(required)* | `(ndof,)` per-joint velocity limit (rad/s or m/s) |
| `max_acceleration` | *(required)* | `(ndof,)` per-joint acceleration limit |
| `method` | `"toppra"` | Backend: `"toppra"` (default) or `"totg"` |
| `max_deviation` | `0.1` | TOTG-only radial blend tolerance at corners. Larger = faster cornering, but the trajectory deviates more from the original waypoints. |
| `time_step` | `1e-3` | TOTG-only forward-integration step along the path arc length. Smaller = more accurate, slower. |
| `velocity_scaling` | `1.0` | Scale factor in `(0, 1]` applied to `max_velocity`. Use to slow the trajectory without changing the stored limits. |
| `acceleration_scaling` | `1.0` | Scale factor in `(0, 1]` applied to `max_acceleration`. |

## Querying the trajectory

The returned `Trajectory` object supports both point and batch queries:

```python
# Point queries at arbitrary time t.
pos = traj.position(t)            # (ndof,)
vel = traj.velocity(t)            # (ndof,)
acc = traj.acceleration(t)        # (ndof,)

# Batch: user-supplied time grid.
pos, vel, acc = traj.sample(times)            # each (T, ndof)

# Batch: uniform grid at controller dt — always includes t=0 and t=duration.
times, pos, vel, acc = traj.sample_uniform(dt=0.01)
```

## Scaling for slower motion

Pass `velocity_scaling` or `acceleration_scaling` to `parameterize()`
to slow the trajectory without reconstructing the parameterizer:

```python
traj_slow = param.parameterize(path, velocity_scaling=0.5)
# traj_slow.duration > traj.duration
```

## One-shot convenience

For scripts where you only parameterize a single path:

```python
from autolife_planning.trajectory import parameterize_path

traj = parameterize_path(path, vel_limits, acc_limits)
```

## Pipeline recipe

A typical end-to-end pipeline:

```
plan(start, goal, simplify=True, interpolate=True)
        │
        ▼
  (N, ndof) path          geometric, no timing
        │
        ▼
  TimeOptimalParameterizer.parameterize(path)
        │
        ▼
  Trajectory               q(t), q̇(t), q̈(t) with bounded vel/acc
        │
        ▼
  traj.sample_uniform(dt)  dense rollout at controller rate
        │
        ▼
  stream to hardware        (times, positions, velocities, accelerations)
```

## API reference

See the full API docs at [Trajectory API](../api/trajectory.md).
