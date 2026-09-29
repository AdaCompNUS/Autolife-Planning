# Subgroup Planning

The Autolife robot exposes 24 joints when you load the full URDF.
Planning over all of them is almost never what you want — for most
tasks, only a slice of the body is under active control while the
rest stays put (or follows its own controller).

**Subgroups are user-defined.** A subgroup is just an ordered list
of indices into the 24-DOF body. The OMPL planner reasons in the
reduced joint space; the C++ collision pipeline injects whatever
you choose as the frozen stance and still checks against the
**full** body. You can define a subgroup over any subset, in any
order — single joint, head + one finger, left arm + one leg, the
mobile base plus one waist pitch, anything.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/subgroup_planning.mp4" type="video/mp4">
</video>

## Built-in aliases

Convenience aliases ship for the common slicings, discoverable via
[`available_robots()`](../api/planning.md#autolife_planning.planning.motion_planner.available_robots):

| Alias | DOF | What it controls |
|---|---:|---|
| `autolife_base` | 3 | mobile base (x, y, yaw) |
| `autolife_height` | 3 | ankle + knee + waist pitch |
| `autolife_left_arm` | 7 | single arm, shoulder → wrist |
| `autolife_right_arm` | 7 | single arm, shoulder → wrist |
| `autolife_dual_arm` | 14 | both arms |
| `autolife_torso_left_arm` | 9 | 2-DOF waist + left arm |
| `autolife_torso_right_arm` | 9 | 2-DOF waist + right arm |
| `autolife_body` | 21 | whole body, base excluded |
| `autolife` | 24 | full body |

They exist as starting points. The list is a dict in
`autolife_planning.autolife.PLANNING_SUBGROUPS` — add your own entry
to extend it, or use the low-level API below for truly one-off
slicings.

## Using a built-in alias

```python
from autolife_planning.autolife import HOME_JOINTS
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

live_config = robot.get_joint_state()          # 24-D array from your system

planner = create_planner(
    "autolife_left_arm",
    config=PlannerConfig(time_limit=1.0),
    base_config=live_config,                    # frozen joints anchored here
    pointcloud=obstacle_cloud,
)

start = planner.extract_config(live_config)     # 24-D → 7-D
goal = planner.sample_valid()
result = planner.plan(start, goal)

full_path = planner.embed_path(result.path)     # 7-D path → 24-D path
```

`base_config` defaults to `HOME_JOINTS`. Pass the live joint reading
instead to pin every non-active joint exactly where the robot
currently is — collision checks see the real stance, not a synthetic
home pose.

## Defining a custom subgroup

Any ordered list of joint indices is a valid subgroup. Drop straight
to the C++ binding when the alias list doesn't match your problem:

```python
from autolife_planning._ompl_vamp import OmplVampPlanner
from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config

# Name the joints you want to plan over; the rest stay frozen.
active_names = [
    "Joint_Waist_Pitch",
    "Joint_Waist_Yaw",
    "Joint_Left_Shoulder_Pitch",
    "Joint_Left_Shoulder_Roll",
    "Joint_Left_Elbow",
]
full_names = autolife_robot_config.joint_names
active_indices = [full_names.index(j) for j in active_names]

planner = OmplVampPlanner(active_indices, HOME_JOINTS.tolist())
planner.add_pointcloud(cloud.tolist(), *planner.min_max_radii(), 0.012)

start = [HOME_JOINTS[i] for i in active_indices]
goal = [...]                                    # your own sampling logic
result = planner.plan(start, goal, planner_name="rrtc", time_limit=1.0)
```

Same planner, same collision pipeline, same point-cloud broadphase
— just a different slice. You can also switch an existing planner
to a different subgroup on the fly with
[`set_subgroup(active_indices, frozen_config)`](../api/planning.md#autolife_planning._ompl_vamp.OmplVampPlanner.set_subgroup),
which preserves the collision environment.

## When it matters

Planning in 24 DOF for a task that only needs an arm wastes almost
all of the search time on irrelevant configurations. For a 7-DOF arm
subgroup on the table-obstacle scene, `rrtc` returns in ~1 ms
median. The same plan over the 24-DOF body runs 2–5× slower for no
practical reason — the base is already where it should be.

Conversely, whole-body subgroups are the right choice when the task
needs the base or legs to move alongside the arm — e.g. the
rls_pick_place demo has the base drive and the legs squat under a
low beam while the arm carries a bowl. Pick the smallest subgroup
that captures the joints your task actually needs to move.
