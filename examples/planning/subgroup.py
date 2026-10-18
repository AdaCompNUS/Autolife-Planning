"""Plan every kinematic subgroup at three different stances.

Demonstrates that the same subgroup name (e.g. ``autolife_left_arm``)
can be planned around any 24-DOF base configuration the caller passes
in — no stance is baked into the planner name.  The three stances
below are *example data*, not part of the planning API.

    pixi run python examples/planning/subgroup.py
"""

import numpy as np
from fire import Fire

from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# Joint values for three example stances.  Replace this dict with any
# 24-DOF array (e.g. the live state from your env) to plan around an
# arbitrary pose.
STANCES = {
    "high": {"Joint_Ankle": 0.0, "Joint_Knee": 0.0, "Joint_Waist_Pitch": 0.00},
    "mid": {"Joint_Ankle": 0.78, "Joint_Knee": 1.60, "Joint_Waist_Pitch": 0.89},
    "low": {"Joint_Ankle": 1.41, "Joint_Knee": 2.38, "Joint_Waist_Pitch": 0.95},
}

SUBGROUPS = [
    "autolife_left_arm",
    "autolife_right_arm",
    "autolife_dual_arm",
    "autolife_torso_left_arm",
    "autolife_torso_right_arm",
    "autolife_height",  # 3 DOF: ankle + knee + waist pitch
    "autolife_base",  # 3 DOF: virtual x, y, yaw
    "autolife_body",  # 21 DOF: legs + waist + arms + neck
]


def base_with_stance(stance: dict[str, float]) -> np.ndarray:
    base = HOME_JOINTS.copy()
    for joint_name, value in stance.items():
        base[autolife_robot_config.joint_names.index(joint_name)] = value
    return base


def plan_and_show(
    env, robot_name: str, base: np.ndarray, config: PlannerConfig, label: str
) -> bool:
    """Plan one subgroup against *base* and animate it interactively.

    Returns ``True`` if the user pressed ``n`` to advance to the next
    demo, ``False`` if the user closed the GUI window (in which case the
    caller should stop iterating).
    """
    planner = create_planner(robot_name, config=config, base_config=base)
    start = planner.extract_config(base)
    goal = planner.sample_valid()

    result = planner.plan(start, goal)
    n_wp = result.path.shape[0] if result.path is not None else 0
    print(f"  [{label}] {result.status.value} — {n_wp} waypoints")

    if result.success and result.path is not None:
        return env.animate_path(planner.embed_path(result.path), next_key="n")
    env.wait_key("n", f"[{label}] no path — press 'n' for next")
    return env.sim.client.isConnected()


def main(planner_name: str = "bitstar", time_limit: float = 0.5):
    """Run the subgroup sweep with the chosen OMPL planner.

    Available planner names (pick one and pass as ``--planner_name``):

        RRT family ........... rrtc / rrtconnect, rrt, rrtstar,
                               informed_rrtstar, rrtsharp, rrtxstatic,
                               lbtrrt, trrt, bitrrt
        Informed trees ....... bitstar, abitstar, aitstar, eitstar, blitstar
        FMT .................. fmt, bfmt
        KPIECE ............... kpiece, bkpiece, lbkpiece
        PRM family ........... prm, prmstar, lazyprm, lazyprmstar,
                               spars, spars2
        Exploration-based .... est, biest, sbl, stride, pdst

    Single-query feasibility planners (``rrtc``, ``rrt``, ``kpiece``,
    ``est``, …) terminate as soon as they find any valid path, usually
    in a few milliseconds.  Asymptotically optimal anytime planners
    (``bitstar``, ``aitstar``, ``rrtstar``, …) keep refining the path
    until ``time_limit`` expires, so they always use the full budget —
    keep ``time_limit`` small for a snappy demo and bump it when you
    care about path quality.
    """
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    config = PlannerConfig(planner_name=planner_name, time_limit=time_limit)

    # Every subgroup × every stance.  Inactive joints are pinned to the
    # stance values; active joints get the stance as their start pose.
    for stance_name, stance in STANCES.items():
        base = base_with_stance(stance)
        for robot_name in SUBGROUPS:
            cont = plan_and_show(
                env, robot_name, base, config, f"{robot_name} @ {stance_name}"
            )
            if not cont:
                return

    env.wait_for_close()


if __name__ == "__main__":
    Fire(main)
