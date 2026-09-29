"""Shared scaffolding for the soft-cost-planning gallery.

Mirrors ``examples/planning/constrained/_shared.py`` one-for-one, with
one deliberate difference: the demos here drive the planner with a
:class:`~autolife_planning.planning.costs.Cost` — a scalar CasADi
expression that RRT*-family planners integrate along every motion —
rather than a hard :class:`Constraint`.  The per-state penalty is
always ``ca.sumsqr(residual)`` of the same residual used in the
constrained demo, so the "manifold" geometry is identical; the
planner just sees it as a preference instead of a law.

We still reuse :meth:`SymbolicContext.project` for goal selection.
Projecting an otherwise-random seed onto ``residual = 0`` gives a
goal that sits exactly on the soft manifold, which makes the cost's
shaping effect visually obvious in the animation (the arm hugs the
manifold whenever it can).
"""

from __future__ import annotations

import numpy as np

from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import SymbolicContext

SUBGROUP = "autolife_left_arm"
EE_LINK = "Link_Left_Gripper"
LEFT_FINGER_LINK = "Link_Left_Gripper_Left_Finger"
RIGHT_FINGER_LINK = "Link_Left_Gripper_Right_Finger"


def ee_translation(ctx: "SymbolicContext"):
    """CasADi expression for the TCP (midpoint between the two fingers)."""
    return 0.5 * (
        ctx.link_translation(LEFT_FINGER_LINK) + ctx.link_translation(RIGHT_FINGER_LINK)
    )


def ee_position(ctx: "SymbolicContext", q: np.ndarray) -> np.ndarray:
    """Numeric evaluation of the TCP at joint config *q*."""
    left = np.asarray(ctx.evaluate_link_pose(LEFT_FINGER_LINK, q))[:3, 3]
    right = np.asarray(ctx.evaluate_link_pose(RIGHT_FINGER_LINK, q))[:3, 3]
    return 0.5 * (left + right)


def setup():
    """Boot a PyBullet env, a SymbolicContext, and the HOME start config."""
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()
    return env, ctx, start


def find_goal(ctx, residual, start, planner, score, n: int = 400, seed: int = 0):
    """Return a valid manifold-feasible goal that maximises ``score(xyz)``.

    Samples random joint perturbations of *start*, projects each onto
    ``residual = 0`` via :meth:`SymbolicContext.project`, and keeps the
    valid config whose gripper position maximises ``score``.  Valid
    means inside the joint bounds AND collision-free under *planner*.

    The cost-planning demos pass the same residual they used to build
    their :class:`Cost` so the goal lands exactly on the soft manifold.
    """
    lower = np.array(planner._planner.lower_bounds())
    upper = np.array(planner._planner.upper_bounds())
    rng = np.random.default_rng(seed)
    best_q, best_s = None, -np.inf
    for _ in range(n):
        seed_q = np.clip(
            start + rng.uniform(-0.7, 0.7, start.shape[0]),
            lower + 0.02,
            upper - 0.02,
        )
        try:
            g = ctx.project(seed_q, residual)
        except RuntimeError:
            continue
        if np.any(g < lower) or np.any(g > upper):
            continue
        if not planner.validate(g):
            continue
        s = float(score(ee_position(ctx, g)))
        if s > best_s:
            best_q, best_s = g, s
    if best_q is None:
        raise RuntimeError("no reachable manifold goal found")
    return best_q


def run_demo(env, planner, start, goal, title: str) -> None:
    """Plan from start to goal, print a one-line summary, animate the path.

    The animation is interactive — ``env.animate_path`` blocks on the
    PyBullet window with space/arrow-key controls and only returns when
    the user closes the viewer.
    """
    print(f"── {title} ──")
    result = planner.plan(start, goal)
    n = result.path.shape[0] if result.path is not None else 0
    print(
        f"  {result.status.value}: {n} waypoints in "
        f"{result.planning_time_ns / 1e6:.0f} ms, cost {result.path_cost:.2f}"
    )
    if result.success and result.path is not None:
        env.animate_path(planner.embed_path(result.path), fps=60)
    else:
        env.wait_for_close()
