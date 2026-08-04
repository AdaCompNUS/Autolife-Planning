"""Abstract base class and unified factory for IK solver backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from autolife_planning.autolife import CHAIN_CONFIGS
from autolife_planning.types import (
    ChainConfig,
    CuroboV2IKConfig,
    IKConfig,
    IKResult,
    PinkIKConfig,
    SE3Pose,
)


class IKSolverBase(ABC):
    """Base class for all IK solvers.

    Concrete implementations (TracIKSolver, PinkIKSolver, CuroboV2IKSolver)
    must implement
    the properties ``base_frame``, ``ee_frame``, ``num_joints`` and
    the methods ``solve()`` and ``fk()``.
    """

    @property
    @abstractmethod
    def base_frame(self) -> str:
        ...

    @property
    @abstractmethod
    def ee_frame(self) -> str:
        ...

    @property
    @abstractmethod
    def num_joints(self) -> int:
        ...

    @abstractmethod
    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config=None,
    ) -> IKResult:
        """Solve inverse kinematics.

        Input:
            target_pose: Desired end-effector pose.
            seed: Initial joint configuration (uses a default if None).
            config: Backend-specific configuration (IKConfig, PinkIKConfig, …).
        Output:
            IKResult with solution status and joint positions.
        """

    @abstractmethod
    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        """Compute forward kinematics for the end effector.

        Input:
            joint_positions: Joint angles array of length ``num_joints``.
        Output:
            SE3Pose of the end effector.
        """


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------


def _resolve_chain_config(
    chain_name: str,
    side: str | None = None,
    urdf_path: str | None = None,
) -> ChainConfig:
    """Look up and optionally patch a chain config."""
    if side is not None:
        chain_name = f"{chain_name}_{side}"

    if chain_name not in CHAIN_CONFIGS:
        available = ", ".join(sorted(CHAIN_CONFIGS.keys()))
        raise ValueError(f"Unknown chain '{chain_name}'. Available chains: {available}")

    chain_config = CHAIN_CONFIGS[chain_name]
    if urdf_path is not None:
        chain_config = chain_config.with_urdf_path(urdf_path)
    return chain_config


def create_ik_solver(
    chain_name: str,
    config: IKConfig | PinkIKConfig | CuroboV2IKConfig | None = None,
    side: str | None = None,
    urdf_path: str | None = None,
    backend: str = "trac_ik",
    *,
    joint_names: list[str] | None = None,
    self_collision: bool = False,
) -> IKSolverBase:
    """Factory function to create an IK solver for a named chain.

    Input:
        chain_name: Name of the chain (e.g. "left_arm", "whole_body").
        config: IK configuration — ``IKConfig`` for trac_ik,
            ``PinkIKConfig`` for pink, or ``CuroboV2IKConfig`` for cuRoboV2
            (uses backend defaults if None).
        side: Optional "left" or "right" suffix for compound names.
        urdf_path: Override the default URDF file path.
        backend: ``"trac_ik"`` (default), ``"pink"``, or ``"curobo_v2"``.
        joint_names: (pink only) Explicit controlled joint names.
            Derived from the kinematic chain if None.
        self_collision: (pink only) Build a collision model from the
            URDF's co-located SRDF for collision avoidance.
    Output:
        An IKSolverBase instance for the requested backend.

    Examples:
        create_ik_solver("left_arm")
        create_ik_solver("whole_body", side="left")
        create_ik_solver("whole_body", side="left", backend="pink",
                         config=PinkIKConfig(com_cost=0.1))
        create_ik_solver("whole_body", side="left", backend="curobo_v2",
                         config=CuroboV2IKConfig())
    """
    chain_config = _resolve_chain_config(chain_name, side, urdf_path)

    if backend == "trac_ik":
        try:
            from autolife_planning.kinematics.trac_ik_solver import TracIKSolver
        except ModuleNotFoundError as exc:
            if exc.name == "pinocchio":
                raise ModuleNotFoundError(
                    "TRAC-IK backend requires 'pinocchio' (PyPI package: 'pin'). "
                    "Install a compatible pin release for your Python version "
                    "(for Python 3.8, pin==2.6.21 is known to work), and run "
                    "with a clean environment (avoid ROS PYTHONPATH/LD_LIBRARY_PATH "
                    "overrides)."
                ) from exc
            raise

        ik_config = config if isinstance(config, IKConfig) else None
        return TracIKSolver(chain_config, ik_config)

    if backend == "pink":
        from autolife_planning.kinematics.collision_model import build_collision_model
        from autolife_planning.kinematics.pink_ik_solver import PinkIKSolver

        pink_config = config if isinstance(config, PinkIKConfig) else None

        collision_context = None
        if self_collision:
            collision_context = build_collision_model(chain_config.urdf_path)

        return PinkIKSolver(
            chain_config,
            config=pink_config,
            joint_names=joint_names,
            collision_context=collision_context,
        )

    if backend in {"curobo_v2", "curobov2", "curobo"}:
        from autolife_planning.kinematics.curobo_v2_ik_solver import (
            CuroboV2IKSolver,
        )

        curobo_config = config if isinstance(config, CuroboV2IKConfig) else None
        return CuroboV2IKSolver(chain_config, config=curobo_config)

    raise ValueError(
        f"Unknown backend '{backend}'. Choose 'trac_ik', 'pink', or 'curobo_v2'."
    )
