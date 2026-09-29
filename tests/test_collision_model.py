"""Self-collision model regression tests.

The SRDF may only disable pairs that cannot collide (see
scripts/audit_collision_pairs.py).  These tests pin the parts of that
contract that matter most in practice: the named poses stay valid, the
hands stay collision-checked against the mobile base (Autolife grasps
from the ground) and against each other, and real contacts between those
body parts are rejected.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import autolife_planning
from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config

pytest.importorskip("autolife_planning._ompl_vamp")

RESOURCES = Path(autolife_planning.__file__).parent / "resources/robot/autolife"
NAMES = list(autolife_robot_config.joint_names)
SQUAT = {"Joint_Ankle": -1.20, "Joint_Knee": -2.40, "Joint_Waist_Pitch": -1.20}

# Body configurations where the only overlapping collision-checked pairs
# are between the two named body parts, and whose CAD meshes intersect.
TOUCHING = {
    "base-left_hand": {
        "Joint_Ankle": -0.644,
        "Joint_Knee": 0.203,
        "Joint_Waist_Pitch": 1.696,
        "Joint_Waist_Yaw": -1.39,
        "Joint_Left_Shoulder_Inner": -0.573,
        "Joint_Left_Shoulder_Outer": -1.275,
        "Joint_Left_UpperArm": -2.496,
        "Joint_Left_Elbow": 0.071,
        "Joint_Left_Forearm": -2.629,
        "Joint_Left_Wrist_Upper": 0.879,
        "Joint_Left_Wrist_Lower": -0.076,
        "Joint_Neck_Roll": -0.244,
        "Joint_Neck_Pitch": -0.287,
        "Joint_Neck_Yaw": 0.098,
        "Joint_Right_Shoulder_Inner": 2.239,
        "Joint_Right_Shoulder_Outer": -0.251,
        "Joint_Right_UpperArm": -1.508,
        "Joint_Right_Elbow": -0.568,
        "Joint_Right_Forearm": 1.905,
        "Joint_Right_Wrist_Upper": 0.99,
        "Joint_Right_Wrist_Lower": -1.221,
    },
    "base-right_hand": {
        "Joint_Ankle": -1.044,
        "Joint_Knee": -2.363,
        "Joint_Waist_Pitch": -0.837,
        "Joint_Waist_Yaw": 0.888,
        "Joint_Left_Shoulder_Inner": 1.138,
        "Joint_Left_Shoulder_Outer": 0.23,
        "Joint_Left_UpperArm": -2.556,
        "Joint_Left_Elbow": 2.177,
        "Joint_Left_Forearm": 1.468,
        "Joint_Left_Wrist_Upper": -0.963,
        "Joint_Left_Wrist_Lower": -1.285,
        "Joint_Neck_Roll": -0.391,
        "Joint_Neck_Pitch": -0.133,
        "Joint_Neck_Yaw": -0.147,
        "Joint_Right_Shoulder_Inner": -2.795,
        "Joint_Right_Shoulder_Outer": 2.343,
        "Joint_Right_UpperArm": 0.921,
        "Joint_Right_Elbow": -1.612,
        "Joint_Right_Forearm": 2.706,
        "Joint_Right_Wrist_Upper": 0.0,
        "Joint_Right_Wrist_Lower": -0.946,
    },
    "left_hand-right_hand": {
        "Joint_Ankle": -1.227,
        "Joint_Knee": -1.866,
        "Joint_Waist_Pitch": -1.013,
        "Joint_Waist_Yaw": 1.097,
        "Joint_Left_Shoulder_Inner": -1.558,
        "Joint_Left_Shoulder_Outer": -2.028,
        "Joint_Left_UpperArm": -2.05,
        "Joint_Left_Elbow": 2.212,
        "Joint_Left_Forearm": -0.397,
        "Joint_Left_Wrist_Upper": -0.297,
        "Joint_Left_Wrist_Lower": 0.101,
        "Joint_Neck_Roll": -0.207,
        "Joint_Neck_Pitch": 0.195,
        "Joint_Neck_Yaw": 0.721,
        "Joint_Right_Shoulder_Inner": -0.908,
        "Joint_Right_Shoulder_Outer": -0.026,
        "Joint_Right_UpperArm": -1.508,
        "Joint_Right_Elbow": -0.987,
        "Joint_Right_Forearm": -2.111,
        "Joint_Right_Wrist_Upper": 0.837,
        "Joint_Right_Wrist_Lower": -0.096,
    },
    "shin-left_hand": {
        "Joint_Ankle": -0.271,
        "Joint_Knee": -0.666,
        "Joint_Waist_Pitch": -1.451,
        "Joint_Waist_Yaw": 2.181,
        "Joint_Left_Shoulder_Inner": 1.76,
        "Joint_Left_Shoulder_Outer": -1.834,
        "Joint_Left_UpperArm": -1.124,
        "Joint_Left_Elbow": 1.661,
        "Joint_Left_Forearm": -0.941,
        "Joint_Left_Wrist_Upper": 0.83,
        "Joint_Left_Wrist_Lower": 0.605,
        "Joint_Neck_Roll": 0.306,
        "Joint_Neck_Pitch": -0.255,
        "Joint_Neck_Yaw": 0.31,
        "Joint_Right_Shoulder_Inner": 0.747,
        "Joint_Right_Shoulder_Outer": 2.615,
        "Joint_Right_UpperArm": 2.019,
        "Joint_Right_Elbow": -0.386,
        "Joint_Right_Forearm": -0.56,
        "Joint_Right_Wrist_Upper": -1.029,
        "Joint_Right_Wrist_Lower": 0.269,
    },
    "head-right_hand": {
        "Joint_Ankle": -0.916,
        "Joint_Knee": -2.317,
        "Joint_Waist_Pitch": -0.662,
        "Joint_Waist_Yaw": -0.455,
        "Joint_Left_Shoulder_Inner": 2.363,
        "Joint_Left_Shoulder_Outer": -2.34,
        "Joint_Left_UpperArm": -1.605,
        "Joint_Left_Elbow": 1.835,
        "Joint_Left_Forearm": -0.856,
        "Joint_Left_Wrist_Upper": 0.208,
        "Joint_Left_Wrist_Lower": -0.763,
        "Joint_Neck_Roll": 0.373,
        "Joint_Neck_Pitch": 0.454,
        "Joint_Neck_Yaw": 0.503,
        "Joint_Right_Shoulder_Inner": -2.637,
        "Joint_Right_Shoulder_Outer": -0.191,
        "Joint_Right_UpperArm": -2.47,
        "Joint_Right_Elbow": -1.687,
        "Joint_Right_Forearm": -1.038,
        "Joint_Right_Wrist_Upper": 0.98,
        "Joint_Right_Wrist_Lower": 0.956,
    },
}


def full(joints):
    q = HOME_JOINTS.copy()
    for name, value in joints.items():
        q[NAMES.index(name)] = value
    return q


@pytest.fixture(scope="module")
def body_planner():
    from autolife_planning.planning import create_planner

    return create_planner("autolife")


@pytest.mark.parametrize(
    "pose",
    [HOME_JOINTS, full(SQUAT), np.zeros(24)],
    ids=["home", "squat", "zero"],
)
def test_named_poses_are_valid(body_planner, pose):
    assert body_planner.validate(pose)


def test_hands_checked_against_base_and_each_other():
    root = ET.parse(RESOURCES / "autolife.srdf").getroot()
    disabled = {
        frozenset((d.get("link1"), d.get("link2")))
        for d in root.iter("disable_collisions")
    }
    links = [
        link.get("name")
        for link in ET.parse(RESOURCES / "autolife_spherized.urdf")
        .getroot()
        .iter("link")
        if link.find("collision/geometry/sphere") is not None
    ]

    def hand(side):
        return [
            n
            for n in links
            if n.startswith(f"Link_{side}_") and ("Gripper" in n or "Wrist" in n)
        ]

    must_check = [("Link_Ground_Vehicle", h) for h in hand("Left") + hand("Right")]
    must_check += [(a, b) for a in hand("Left") for b in hand("Right")]
    assert len(must_check) == 99
    assert not [p for p in must_check if frozenset(p) in disabled]


@pytest.mark.parametrize("contact", TOUCHING)
def test_real_contacts_are_rejected(body_planner, contact):
    assert not body_planner.validate(full(TOUCHING[contact]))
