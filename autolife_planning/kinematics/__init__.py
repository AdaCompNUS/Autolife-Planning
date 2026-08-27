# IK solver base + unified factory (always available)
from .ik_solver_base import IKSolverBase, create_ik_solver

__all__ = [
    "IKSolverBase",
    "create_ik_solver",
]

# cuRoboV2 imports its optional CUDA/PyTorch runtime lazily, so exposing the
# class here does not make the base package depend on those libraries.
from .curobo_v2_ik_solver import CuroboV2IKSolver, CuroboV2UnavailableError

__all__ += ["CuroboV2IKSolver", "CuroboV2UnavailableError"]

# TRAC-IK backend also depends on pinocchio for world/base transform mapping.
try:
    from .trac_ik_solver import TracIKSolver

    __all__ += ["TracIKSolver"]
except (ModuleNotFoundError, ImportError):
    # pinocchio not installed — TracIKSolver unavailable.
    pass

# Additional Pinocchio-dependent modules (pink, collision, FK helpers).
try:
    from .collision_model import (
        CollisionContext,
        add_pointcloud_obstacles,
        build_collision_model,
    )
    from .pink_ik_solver import PinkIKSolver
    from .pinocchio_fk import (
        PinocchioContext,
        compute_forward_kinematics,
        compute_jacobian,
        create_pinocchio_context,
    )

    __all__ += [
        "PinocchioContext",
        "create_pinocchio_context",
        "compute_forward_kinematics",
        "compute_jacobian",
        "CollisionContext",
        "build_collision_model",
        "add_pointcloud_obstacles",
        "PinkIKSolver",
    ]
except (ModuleNotFoundError, ImportError):
    # pinocchio (or pink) not installed — pinocchio FK, collision model,
    # and PinkIKSolver are unavailable.
    pass
