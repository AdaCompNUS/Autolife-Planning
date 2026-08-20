"""cuRoboV2 cost-manager extensions used by constrained Autolife IK.

This module imports the optional cuRobo/PyTorch backend and must therefore only
be imported lazily while constructing a constrained cuRobo context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.metrics import CostCollection


class WaistAnkleConstraint(BaseCost):
    """GPU inequality constraint on waist pitch relative to the ankle."""

    def __init__(
        self,
        config: WaistAnkleConstraintCfg,
        ankle_index: int,
        waist_index: int,
    ) -> None:
        super().__init__(config)
        self._ankle_index = ankle_index
        self._waist_index = waist_index
        self._lower_bound = torch.as_tensor(
            config.lower_bound,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self._upper_bound = torch.as_tensor(
            config.upper_bound,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

    def forward(self, joint_position: torch.Tensor) -> torch.Tensor:
        """Return zero inside the interval and a positive hinge outside it."""
        difference = (
            joint_position[..., self._waist_index]
            - joint_position[..., self._ankle_index]
        )
        lower_violation = torch.clamp_min(self._lower_bound - difference, 0.0)
        upper_violation = torch.clamp_min(difference - self._upper_bound, 0.0)
        violation = self._weight[0] * (lower_violation + upper_violation)
        return violation.unsqueeze(-1)


@dataclass
class WaistAnkleConstraintCfg(BaseCostCfg):
    """Configuration for :class:`WaistAnkleConstraint`."""

    lower_bound: float = 0.0
    upper_bound: float = 0.0
    ankle_joint_name: str = "Joint_Ankle"
    waist_joint_name: str = "Joint_Waist_Pitch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lower_bound > self.upper_bound:
            raise ValueError("waist-ankle lower_bound must be <= upper_bound")


@dataclass
class AutolifeRobotCostManagerCfg(RobotCostManagerCfg):
    """Standard cuRobo costs plus an optional waist-ankle constraint."""

    waist_ankle_cfg: WaistAnkleConstraintCfg | None = None

    def __post_init__(self) -> None:
        self.class_type = AutolifeRobotCostManager

    @staticmethod
    def create(
        data_dict: dict[str, Any],
        scene_collision_checker: Any = None,
        device_cfg: Any = None,
    ) -> AutolifeRobotCostManagerCfg:
        base = RobotCostManagerCfg.create(
            data_dict,
            scene_collision_checker=scene_collision_checker,
            device_cfg=device_cfg,
        )
        waist_ankle_data = data_dict.get("waist_ankle_cfg")
        waist_ankle_cfg = None
        if waist_ankle_data is not None:
            waist_ankle_cfg = WaistAnkleConstraintCfg(
                **waist_ankle_data,
                device_cfg=device_cfg,
            )
        return AutolifeRobotCostManagerCfg(
            self_collision_cfg=base.self_collision_cfg,
            scene_collision_cfg=base.scene_collision_cfg,
            cspace_cfg=base.cspace_cfg,
            start_cspace_dist_cfg=base.start_cspace_dist_cfg,
            target_cspace_dist_cfg=base.target_cspace_dist_cfg,
            tool_pose_cfg=base.tool_pose_cfg,
            waist_ankle_cfg=waist_ankle_cfg,
        )


class AutolifeRobotCostManager(RobotCostManager):
    """cuRobo cost manager that evaluates the waist-ankle constraint."""

    def initialize_from_config(
        self,
        config: AutolifeRobotCostManagerCfg,
        transition_model: Any,
        scene_collision_checker: Any = None,
        **kwargs: Any,
    ) -> None:
        super().initialize_from_config(
            config,
            transition_model,
            scene_collision_checker,
            **kwargs,
        )
        if config.waist_ankle_cfg is None:
            return

        joint_names = list(transition_model.joint_names)
        ankle_name = config.waist_ankle_cfg.ankle_joint_name
        waist_name = config.waist_ankle_cfg.waist_joint_name
        try:
            ankle_index = joint_names.index(ankle_name)
            waist_index = joint_names.index(waist_name)
        except ValueError as exc:
            raise ValueError(
                "waist-ankle constraint joints are not cuRobo optimization DOFs: "
                f"{ankle_name!r}, {waist_name!r}; available joints: {joint_names}"
            ) from exc
        # Execute this tiny tensor-only constraint on the rollout's current
        # stream. Registering a separate cost stream would make the base
        # manager wait on an event before this extension records it, which is
        # invalid during CUDA-graph capture.
        self.costs["waist_ankle"] = WaistAnkleConstraint(
            config.waist_ankle_cfg,
            ankle_index=ankle_index,
            waist_index=waist_index,
        )

    def compute_costs(
        self,
        state: Any,
        cost_collection: CostCollection | None = None,
        goal: Any = None,
        **kwargs: Any,
    ) -> CostCollection:
        collection = super().compute_costs(
            state,
            cost_collection=cost_collection,
            goal=goal,
            **kwargs,
        )
        if self.has_cost("waist_ankle"):
            constraint = self.get_cost("waist_ankle")
            if constraint.enabled:
                value = constraint.forward(state.joint_state.position)
                collection.add(value, "waist_ankle")
        return collection


def add_waist_ankle_constraint(
    solver_config: Any,
    *,
    lower_bound: float,
    upper_bound: float,
    weight: float,
) -> None:
    """Attach the constraint before cuRobo constructs its rollout managers."""
    rollout_configs = [
        *solver_config.core_cfg.optimizer_rollout_configs,
        solver_config.core_cfg.metrics_rollout_config,
    ]
    for rollout in rollout_configs:
        manager_cfg = rollout.constraint_cfg
        if not isinstance(manager_cfg, AutolifeRobotCostManagerCfg):
            raise TypeError("cuRobo did not construct the Autolife constraint manager")
        manager_cfg.waist_ankle_cfg = WaistAnkleConstraintCfg(
            weight=float(weight),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            device_cfg=rollout.device_cfg,
        )
