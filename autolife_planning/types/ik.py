from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class IKStatus(Enum):
    """Status of IK solution attempt."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    SINGULAR = "singular"
    FAILED = "failed"


class SolveType(Enum):
    """TRAC-IK solve type.

    SPEED:    Return first valid solution (fastest).
    DISTANCE: Minimize joint displacement from seed configuration.
    MANIP1:   Maximize manipulability (product of Jacobian singular values).
    MANIP2:   Maximize isotropy (min/max Jacobian singular value ratio).
    """

    SPEED = "Speed"
    DISTANCE = "Distance"
    MANIP1 = "Manip1"
    MANIP2 = "Manip2"


@dataclass
class IKConfig:
    """Configuration parameters for TRAC-IK solver."""

    timeout: float = 0.2  # Seconds for TRAC-IK dual-thread solve
    epsilon: float = 1e-5  # TRAC-IK convergence tolerance
    solve_type: SolveType = SolveType.SPEED  # Solve strategy
    max_attempts: int = 10  # Python-level random restart attempts
    position_tolerance: float = 1e-4  # Post-solve position validation (meters)
    orientation_tolerance: float = 1e-4  # Post-solve orientation validation (radians)

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.position_tolerance <= 0:
            raise ValueError("position_tolerance must be > 0")
        if self.orientation_tolerance <= 0:
            raise ValueError("orientation_tolerance must be > 0")


@dataclass
class IKResult:
    """Result of an IK solution attempt."""

    status: IKStatus
    joint_positions: np.ndarray | None  # Solution if successful
    final_error: float  # Final pose error
    iterations: int  # Number of iterations performed
    position_error: float  # Final position error (meters)
    orientation_error: float  # Final orientation error (radians)

    @property
    def success(self) -> bool:
        return self.status == IKStatus.SUCCESS


# ---------------------------------------------------------------------------
# Pink constrained IK types
# ---------------------------------------------------------------------------


@dataclass
class CoupledJoint:
    """Linear coupling constraint: ``slave = multiplier * master + offset``.

    Enforced as a hard constraint on both the velocity and position level
    during the Pink IK iterative solve.
    """

    master: str
    slave: str
    multiplier: float = 2.0
    offset: float = 0.0


@dataclass
class PinkIKConfig:
    """Configuration for the Pink constrained IK solver.

    Attributes:
        dt: Integration timestep for differential IK (seconds).
        max_iterations: Maximum solver iterations before returning.
        convergence_thresh: Position convergence threshold (meters).
        orientation_thresh: Orientation convergence threshold (radians).
        position_cost: End-effector position tracking weight.
        orientation_cost: End-effector orientation tracking weight.
        lm_damping: Levenberg-Marquardt damping for singularity avoidance.
        posture_cost: Posture regularization weight (toward seed).
        com_cost: Center-of-mass stability weight (0 disables).
            Penalizes CoM drift from the seed configuration to prevent tipping.
        camera_frame: Frame name to stabilize (e.g. "Link_Camera_Head_Forehead").
            When set with camera_cost > 0, adds a FrameTask that anchors this
            frame to its pose at the seed configuration, keeping the camera
            observation stable while the arm moves.
        camera_cost: Weight for the camera stability task (0 disables).
        coupled_joints: Linear joint coupling constraints enforced as hard
            constraints.  Each entry locks ``slave = multiplier * master + offset``
            on every iteration, replacing inaccurate CoM modelling with a
            kinematic heuristic.
        self_collision: Enable collision barrier (self-collision + obstacles).
        collision_pairs: Number of closest collision pairs to evaluate per step.
        collision_gain: Barrier gain — higher means harder repulsion.
        collision_d_min: Minimum allowed distance between collision pairs (m).
        solver: QP solver backend for qpsolvers (e.g. "osqp", "quadprog").
    """

    dt: float = 0.01
    max_iterations: int = 300
    convergence_thresh: float = 1e-3
    orientation_thresh: float = 1e-2
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 1e-3
    posture_cost: float = 1e-3
    com_cost: float = 0.0
    camera_frame: str | None = None
    camera_cost: float = 0.0
    coupled_joints: list[CoupledJoint] = field(default_factory=list)
    self_collision: bool = False
    collision_pairs: int = 3
    collision_gain: float = 1.0
    collision_d_min: float = 0.02
    solver: str = "osqp"

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.lm_damping < 0:
            raise ValueError("lm_damping must be >= 0")
        if self.collision_pairs < 1:
            raise ValueError("collision_pairs must be >= 1")
        if self.collision_d_min < 0:
            raise ValueError("collision_d_min must be >= 0")


@dataclass
class ConstrainedIKResult:
    """Result of a constrained IK solve, with trajectory from seed to solution.

    Shares fields with :class:`IKResult`; adds ``trajectory``.
    """

    status: IKStatus
    joint_positions: np.ndarray | None
    final_error: float
    iterations: int
    position_error: float
    orientation_error: float
    trajectory: np.ndarray | None = None  # (N, n_joints)

    @property
    def success(self) -> bool:
        return self.status == IKStatus.SUCCESS
