"""The Autolife robot's bundled description.

This module collects every concrete value that describes the one
robot this project ships: joint groupings, the home pose, the URDF
chains TRAC-IK and Pinocchio operate on, the VAMP planning subgroups,
and the top-level ``RobotConfig`` instance.

The dataclass *types* themselves live in :mod:`autolife_planning.types.robot`
— this file holds *values* of those types.
"""

from __future__ import annotations

import os

import numpy as np

from autolife_planning.types.robot import CameraConfig, ChainConfig, RobotConfig

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "autolife")

# Atomic joint groups — indices into the full 24-DOF configuration array.
# Order must match VAMP's URDF tree traversal: base, legs, waist, left arm, neck, right arm.
JOINT_GROUPS = {
    "base": slice(0, 3),  # Virtual_X, Virtual_Y, Virtual_Theta
    "legs": slice(3, 5),  # Ankle, Knee
    "waist": slice(5, 7),  # Waist_Pitch, Waist_Yaw
    "left_arm": slice(7, 14),  # Shoulder → Wrist (7 DOF)
    "neck": slice(14, 17),  # Roll, Pitch, Yaw
    "right_arm": slice(17, 24),  # Shoulder → Wrist (7 DOF)
}

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    "left_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "right_arm": ChainConfig(
        base_link="Link_Waist_Yaw_to_Shoulder_Inner",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_left": ChainConfig(
        base_link="Link_Ground_Vehicle",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_right": ChainConfig(
        base_link="Link_Ground_Vehicle",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    ),
    "whole_body_base_left": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Left_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
    "whole_body_base_right": ChainConfig(
        base_link="Link_Zero_Point",
        ee_link="Link_Right_Wrist_Lower_to_Gripper",
        num_joints=14,
        urdf_path=os.path.join(_RESOURCES_DIR, "autolife_base.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "autolife_viz.urdf")

# Conservative placeholder limits for time-optimal trajectory generation.
# Uniform across all 24 joints until real per-joint specs
# are wired in — the URDFs currently carry a ``velocity="1"`` placeholder on
# every joint and no acceleration field at all, so these values are the
# practical source of truth for the example scripts.
#
# Override these per-joint when real robot specs are available, e.g. faster
# wrists and slower legs/ankle/knee.
MAX_VELOCITY = np.full(24, 0.5, dtype=np.float64)  # rad/s
MAX_ACCELERATION = np.full(24, 0.6, dtype=np.float64)  # rad/s^2

autolife_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "autolife.urdf"),
    joint_names=[
        # [0:3]   base (virtual planar joints)
        "Joint_Virtual_X",
        "Joint_Virtual_Y",
        "Joint_Virtual_Theta",
        # [3:5]   legs
        "Joint_Ankle",
        "Joint_Knee",
        # [5:7]   waist
        "Joint_Waist_Pitch",
        "Joint_Waist_Yaw",
        # [7:14]  left arm
        "Joint_Left_Shoulder_Inner",
        "Joint_Left_Shoulder_Outer",
        "Joint_Left_UpperArm",
        "Joint_Left_Elbow",
        "Joint_Left_Forearm",
        "Joint_Left_Wrist_Upper",
        "Joint_Left_Wrist_Lower",
        # [14:17] neck
        "Joint_Neck_Roll",
        "Joint_Neck_Pitch",
        "Joint_Neck_Yaw",
        # [17:24] right arm
        "Joint_Right_Shoulder_Inner",
        "Joint_Right_Shoulder_Outer",
        "Joint_Right_UpperArm",
        "Joint_Right_Elbow",
        "Joint_Right_Forearm",
        "Joint_Right_Wrist_Upper",
        "Joint_Right_Wrist_Lower",
    ],
    camera=CameraConfig(
        link_name="Link_Camera_Head_Forehead",
        width=640,
        height=480,
        fov=60.0,
        near=0.1,
        far=10.0,
    ),
    max_velocity=MAX_VELOCITY,
    max_acceleration=MAX_ACCELERATION,
)

# VAMP subgroup robot names for planning.
# Each maps to a separate VAMP module with the full robot body (all links and
# collision geometry) but only the listed joints movable.
_LEFT_ARM_JOINTS = [
    "Joint_Left_Shoulder_Inner",
    "Joint_Left_Shoulder_Outer",
    "Joint_Left_UpperArm",
    "Joint_Left_Elbow",
    "Joint_Left_Forearm",
    "Joint_Left_Wrist_Upper",
    "Joint_Left_Wrist_Lower",
]
_RIGHT_ARM_JOINTS = [
    "Joint_Right_Shoulder_Inner",
    "Joint_Right_Shoulder_Outer",
    "Joint_Right_UpperArm",
    "Joint_Right_Elbow",
    "Joint_Right_Forearm",
    "Joint_Right_Wrist_Upper",
    "Joint_Right_Wrist_Lower",
]
_WAIST_JOINTS = ["Joint_Waist_Pitch", "Joint_Waist_Yaw"]
_BASE_JOINTS = ["Joint_Virtual_X", "Joint_Virtual_Y", "Joint_Virtual_Theta"]
# Sagittal chain that bends the robot up/down — used to plan height changes.
_HEIGHT_JOINTS = ["Joint_Ankle", "Joint_Knee", "Joint_Waist_Pitch"]
_LEGS_JOINTS = ["Joint_Ankle", "Joint_Knee"]
PLANNING_SUBGROUPS = {
    # Mobile base in the ground plane (3 DOF: x, y, yaw)
    "autolife_base": {"dof": 3, "joints": _BASE_JOINTS},
    # Height chain (3 DOF: ankle + knee + waist pitch)
    "autolife_height": {"dof": 3, "joints": _HEIGHT_JOINTS},
    # Single arm (7 DOF)
    "autolife_left_arm": {"dof": 7, "joints": _LEFT_ARM_JOINTS},
    "autolife_right_arm": {"dof": 7, "joints": _RIGHT_ARM_JOINTS},
    # Dual arm (14 DOF)
    "autolife_dual_arm": {"dof": 14, "joints": _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS},
    # Torso + arm (9 DOF: 2 waist + 7 arm)
    "autolife_torso_left_arm": {"dof": 9, "joints": _WAIST_JOINTS + _LEFT_ARM_JOINTS},
    "autolife_torso_right_arm": {
        "dof": 9,
        "joints": _WAIST_JOINTS + _RIGHT_ARM_JOINTS,
    },
    # Torso + dual arm (16 DOF: 2 waist + 7 left arm + 7 right arm)
    "autolife_torso_dual_arm": {
        "dof": 16,
        "joints": _WAIST_JOINTS + _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
    },
    # Legs + torso + dual arm (18 DOF: 2 legs + 2 waist + 7 left arm + 7 right arm)
    "autolife_leg_torso_dual_arm": {
        "dof": 18,
        "joints": _LEGS_JOINTS + _WAIST_JOINTS + _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS,
    },
    # Whole body without base (21 DOF)
    "autolife_body": {
        "dof": 21,
        "joints": [
            "Joint_Ankle",
            "Joint_Knee",
            *_WAIST_JOINTS,
            *_LEFT_ARM_JOINTS,
            "Joint_Neck_Roll",
            "Joint_Neck_Pitch",
            "Joint_Neck_Yaw",
            *_RIGHT_ARM_JOINTS,
        ],
    },
}

HOME_JOINTS = np.array(
    [
        # [0:3]   base
        0.0,
        0.0,
        0.0,
        # [3:5]   legs
        0.0,
        0.0,
        # [5:7]   waist
        0.00,
        -0.14,
        # [7:14]  left arm
        0.70,
        -0.14,
        -0.09,
        2.31,
        0.04,
        -0.40,
        0.0,
        # [14:17] neck
        0.0,
        0.0,
        0.0,
        # [17:24] right arm (symmetric with left)
        -0.70,
        0.14,
        -0.09,
        -2.31,
        -0.04,
        -0.40,
        0.0,
    ]
)
