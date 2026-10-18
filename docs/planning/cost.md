# Cost-space Planning

The soft counterpart to [manifold planning](manifold.md). Instead
of forcing every sampled state onto a constraint set, you turn the
same residual into a **preference**: the planner is still free to
leave the manifold, but doing so costs more. An
asymptotically-optimal planner (`rrtstar`, `bitstar`, `aitstar`,
`eitstar`, `fmt`) then refines the path toward the soft manifold
whenever that's the cheaper option.

Every cost is defined inline by the caller as a scalar CasADi
expression in the active joint vector. The library takes the
symbolic gradient, compiles to a `.so`, caches it, and hands it
to the C++ planner, which wraps it as an
`ompl::StateCostIntegralObjective` — trapezoidally integrating the
per-state cost along every edge.

Formally, OMPL minimises the line-integral path cost

$$
J[\pi] \;=\; \int_0^1 c\bigl(\pi(s)\bigr)\, \bigl\lVert \dot{\pi}(s) \bigr\rVert \, ds
$$

where \(c(q)\) is the scalar you write below. The costs replace
OMPL's default path-length objective (no length term is added), so
a path pays only where \(c > 0\).

Four soft counterparts to the manifold examples are laid out
below — the math on the left, the exact code on the right.

## Plane

Pull the gripper toward its home height \(z_0\):

<div class="grid" markdown>

=== "Math"

    $$
    c(q) \;=\; w \cdot \bigl(p_{\text{tcp},z}(q) - z_0\bigr)^2
    $$

    One squared residual times a weight.  The planner is free to
    leave the plane, but paying a quadratic penalty proportional to
    the deviation squared.

=== "Code"

    ```python
    p0 = ee_position(ctx, start)

    residual = ee_translation(ctx)[2] - float(p0[2])
    plane_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="plane_z_cost",
        weight=50.0,
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/cost_plane.mp4" type="video/mp4">
</video>

## Horizontal rail

Prefer a 1-D line in world \(x\) with the orientation frozen:

<div class="grid" markdown>

=== "Math"

    $$
    c(q) \;=\; w \cdot \Bigl\lVert
    \begin{bmatrix}
      p_{\text{tcp},y}(q) - y_0 \\[2pt]
      p_{\text{tcp},z}(q) - z_0 \\[2pt]
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix}
    \Bigr\rVert^2
    $$

    Same 8-row residual as the hard rail constraint, now summed up
    as one quadratic penalty.

=== "Code"

    ```python
    p0 = ee_position(ctx, start)
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    tcp = ee_translation(ctx)
    rot = ctx.link_rotation(EE_LINK)

    residual = ca.vertcat(
        tcp[1] - float(p0[1]),
        tcp[2] - float(p0[2]),
        rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    line_h_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="line_h_cost",
        weight=50.0,
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/cost_line_horizontal.mp4" type="video/mp4">
</video>

## Vertical rail

Soft vertical rail — same pattern, now along world \(z\):

<div class="grid" markdown>

=== "Math"

    $$
    c(q) \;=\; w \cdot \Bigl\lVert
    \begin{bmatrix}
      p_{\text{tcp},x}(q) - x_0 \\[2pt]
      p_{\text{tcp},y}(q) - y_0 \\[2pt]
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix}
    \Bigr\rVert^2
    $$

=== "Code"

    ```python
    p0 = ee_position(ctx, start)
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    tcp = ee_translation(ctx)
    rot = ctx.link_rotation(EE_LINK)

    residual = ca.vertcat(
        tcp[0] - float(p0[0]),
        tcp[1] - float(p0[1]),
        rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    line_v_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="line_v_cost",
        weight=50.0,
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/cost_line_vertical.mp4" type="video/mp4">
</video>

## Orientation lock

Let the gripper translate freely but prefer the home orientation:

<div class="grid" markdown>

=== "Math"

    $$
    c(q) \;=\; w \cdot \Bigl\lVert
    \begin{bmatrix}
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix}
    \Bigr\rVert^2
    $$

=== "Code"

    ```python
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    rot = ctx.link_rotation(EE_LINK)

    residual = ca.vertcat(
        rot[:, 0] - ca.DM(R0[:, 0].tolist()),
        rot[:, 1] - ca.DM(R0[:, 1].tolist()),
    )
    orient_cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="orient_lock_cost",
        weight=50.0,
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/cost_orientation_lock.mp4" type="video/mp4">
</video>

## How to plug it into a planner

```python
planner = create_planner(
    "autolife_left_arm",
    config=PlannerConfig(
        planner_name="rrtstar",       # any AO planner: bitstar, aitstar, eitstar, fmt, ...
        time_limit=5.0,
        simplify=False,               # the default shortcutter ignores the custom cost
    ),
    costs=[plane_cost],               # or multiple: [plane, orient_soft, ...]
)

goal = planner.sample_valid()         # no manifold projection required
result = planner.plan(start, goal)
```

Multiple costs are summed with their weights. Because the cost is a
preference rather than a constraint, **there's no projection step**:
the start and goal can be anywhere the validator accepts, not only
on a manifold.

## Constraint or cost — which to pick?

| | Hard constraint | Soft cost |
|---|---|---|
| Guaranteed to satisfy | yes — every sampled state | no — only preferred |
| Solver | single-tree planners on the projected state space (`rrtc`, `rrt`, `rrtstar`, `prm`, `kpiece`, …; BIT\*, AIT\*, FMT\*, … are rejected) | AO planners only (`rrtstar`, `bitstar`, …) |
| Start / goal | must lie on the manifold | anywhere valid |
| Typical use | end-effector *must* be on a rail / plane | prefer a pose, but let the planner deviate when forced |
| Runtime overhead | projection on every sample | extra motion-cost integral |

Start with a hard constraint when physics demands it (e.g., the
gripper is *literally* riding a rail). Switch to a soft cost when
the preference is just strong but finite — the arm should hug the
shelf when it can, but don't make the task infeasible if it has to
duck.
