"""CUDA-accelerated cuRoboV2 inverse kinematics.

The backend is optional and imported lazily.  Standard :meth:`solve` and
:meth:`solve_batch` use the original URDF, while their constrained variants use
an Autolife stability model that couples the knee to the ankle and bounds the
independent waist pitch relative to the ankle.  Collision costs are not added
to cuRobo.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from autolife_planning.kinematics.ik_solver_base import IKSolverBase
from autolife_planning.types import (
    ChainConfig,
    ConstrainedIKResult,
    CuroboV2IKConfig,
    IKResult,
    IKStatus,
    SE3Pose,
)

_ANKLE = "Joint_Ankle"
_KNEE = "Joint_Knee"
_WAIST_PITCH = "Joint_Waist_Pitch"
_STABILITY_JOINTS = {_ANKLE, _KNEE, _WAIST_PITCH}
_STABLE_URDF_SCHEMA = 3


class CuroboV2UnavailableError(ImportError):
    """Raised when the optional cuRoboV2 runtime is not installed."""


@dataclass
class _CuroboContext:
    solver: Any
    pose_type: type
    goal_tool_pose_type: type
    joint_state_type: type
    device_cfg: Any
    joint_names: tuple[str, ...]
    constrained: bool


def _as_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _frame_parent_joint(frame: Any) -> int:
    if hasattr(frame, "parentJoint"):
        return int(frame.parentJoint)
    if hasattr(frame, "parent"):
        return int(frame.parent)
    raise AttributeError("Unsupported Pinocchio Frame API")


def _chain_joint_ids(model: Any, base_frame: str, ee_frame: str) -> list[int]:
    if not model.existFrame(base_frame):
        raise ValueError(f"Base frame '{base_frame}' not found in URDF")
    if not model.existFrame(ee_frame):
        raise ValueError(f"End-effector frame '{ee_frame}' not found in URDF")

    current = _frame_parent_joint(model.frames[model.getFrameId(ee_frame)])
    base_joint = _frame_parent_joint(model.frames[model.getFrameId(base_frame)])
    joint_ids: list[int] = []
    while current != base_joint and current > 0:
        joint_ids.append(current)
        current = int(model.parents[current])
    if current != base_joint:
        raise ValueError(f"'{ee_frame}' is not downstream of '{base_frame}'")
    joint_ids.reverse()
    return joint_ids


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    joint = next(
        (item for item in root.findall("joint") if item.attrib.get("name") == name),
        None,
    )
    if joint is None:
        raise ValueError(f"Stability joint '{name}' is missing from the URDF")
    return joint


def _set_limit(joint: ET.Element, lower: float, upper: float) -> None:
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"Joint '{joint.attrib.get('name')}' has no limits")
    limit.set("lower", str(float(lower)))
    limit.set("upper", str(float(upper)))


def _set_mimic(
    joint: ET.Element,
    master: str,
    multiplier: float,
    offset: float = 0.0,
) -> None:
    old = joint.find("mimic")
    if old is not None:
        joint.remove(old)
    ET.SubElement(
        joint,
        "mimic",
        {
            "joint": master,
            "multiplier": str(float(multiplier)),
            "offset": str(float(offset)),
        },
    )


def _stable_urdf_path(source_urdf: str, config: CuroboV2IKConfig) -> str:
    """Return a cached URDF with the hard Autolife leg relation."""
    source = Path(source_urdf).resolve()
    payload = (
        f"{_STABLE_URDF_SCHEMA}:{source}:{source.stat().st_mtime_ns}:"
        f"{config.stability_ankle_min}:{config.stability_ankle_max}:"
        f"{config.knee_multiplier}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "autolife_planning_curobo_v2" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{source.stem}_stable.urdf"

    if not output.is_file():
        root = ET.fromstring(source.read_bytes())
        ankle = _find_joint(root, _ANKLE)
        knee = _find_joint(root, _KNEE)

        ankle_min = float(config.stability_ankle_min)
        ankle_max = float(config.stability_ankle_max)
        _set_limit(ankle, ankle_min, ankle_max)
        _set_limit(
            knee,
            config.knee_multiplier * ankle_min,
            config.knee_multiplier * ankle_max,
        )
        _set_mimic(knee, _ANKLE, config.knee_multiplier)

        # A mimic joint must be allowed to move at least multiplier times as
        # fast as its master, otherwise cuRobo tightens the master's velocity.
        ankle_limit = ankle.find("limit")
        ankle_velocity = (
            float(ankle_limit.attrib["velocity"])
            if ankle_limit is not None and "velocity" in ankle_limit.attrib
            else None
        )
        if ankle_velocity is not None:
            knee.find("limit").set(
                "velocity", str(abs(config.knee_multiplier) * ankle_velocity)
            )

        staging = output.with_name(f"{output.name}.{os.getpid()}.partial")
        ET.ElementTree(root).write(staging, encoding="utf-8", xml_declaration=True)
        os.replace(staging, output)

    # cuRobo resolves package://meshes relative to the URDF asset directory.
    source_meshes = source.parent / "meshes"
    cached_meshes = cache_dir / "meshes"
    if source_meshes.is_dir() and not cached_meshes.exists():
        try:
            cached_meshes.symlink_to(source_meshes, target_is_directory=True)
        except FileExistsError:
            pass
    return str(output)


class CuroboV2IKSolver(IKSolverBase):
    """cuRoboV2 IK with an optional hard stability-constrained solve.

    Targets and FK poses use the URDF root/world frame, matching the other
    package IK backends.  cuRobo itself works in ``base_frame`` coordinates;
    the fixed root-to-base transform is handled internally.  Collision costs
    and collision spheres are deliberately disabled.
    """

    def __init__(
        self,
        chain_config: ChainConfig,
        config: CuroboV2IKConfig | None = None,
    ) -> None:
        self._chain_config = chain_config
        self._config = config or CuroboV2IKConfig()
        self._contexts: dict[tuple[bool, CuroboV2IKConfig], _CuroboContext] = {}

        try:
            pin = importlib.import_module("pinocchio")
        except ModuleNotFoundError as exc:
            raise CuroboV2UnavailableError(
                "cuRoboV2 IK requires Pinocchio for package-consistent FK."
            ) from exc
        self._pin = pin
        self._model = pin.buildModelFromUrdf(chain_config.urdf_path)
        self._data = self._model.createData()
        self._joint_ids = _chain_joint_ids(
            self._model, chain_config.base_link, chain_config.ee_link
        )
        self._joint_names = [str(self._model.names[jid]) for jid in self._joint_ids]
        self._name_to_public = {
            name: index for index, name in enumerate(self._joint_names)
        }
        self._ee_frame_id = self._model.getFrameId(chain_config.ee_link)
        self._stability_supported = _STABILITY_JOINTS.issubset(self._joint_names)

        q_neutral = pin.neutral(self._model)
        pin.forwardKinematics(self._model, self._data, q_neutral)
        pin.updateFramePlacements(self._model, self._data)
        base_pose = self._data.oMf[self._model.getFrameId(chain_config.base_link)]
        self._base_rotation = np.asarray(base_pose.rotation, dtype=np.float64).copy()
        self._base_translation = np.asarray(
            base_pose.translation, dtype=np.float64
        ).copy()

    @property
    def base_frame(self) -> str:
        return self._chain_config.base_link

    @property
    def ee_frame(self) -> str:
        return self._chain_config.ee_link

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def stability_supported(self) -> bool:
        """Whether this chain contains the Autolife leg stability joints."""
        return self._stability_supported

    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        q = self._to_full_q(self._validate_seed(joint_positions))
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        pose = self._data.oMf[self._ee_frame_id]
        return SE3Pose(
            position=np.asarray(pose.translation, dtype=np.float64).copy(),
            rotation=np.asarray(pose.rotation, dtype=np.float64).copy(),
        )

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: CuroboV2IKConfig | None = None,
    ) -> IKResult:
        """Solve ordinary cuRobo IK without collision checking."""
        cfg = config or self._config
        batch_seed = None
        if seed is not None:
            batch_seed = self._validate_seed(seed)[None, :]
        return self.solve_batch([target_pose], batch_seed, cfg)[0]

    def solve_batch(
        self,
        target_poses: Sequence[SE3Pose],
        seeds: np.ndarray | None = None,
        config: CuroboV2IKConfig | None = None,
    ) -> list[IKResult]:
        """Solve multiple IK targets in true cuRobo GPU batches.

        Results preserve input order and contain one best solution (nearest
        the corresponding seed) or one failure per target.  Calls larger than
        ``max_batch_size`` are chunked.  Short chunks are padded internally so
        a CUDA-graph context always executes at its configured fixed size.
        """
        cfg = config or self._config
        poses = self._validate_target_poses(target_poses)
        batch_seeds = self._validate_batch_seeds(seeds, len(poses))
        if not poses:
            return []
        return self._solve_batch_impl(
            poses,
            batch_seeds,
            cfg,
            constrained=False,
        )

    def solve_constrained(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: CuroboV2IKConfig | None = None,
    ) -> ConstrainedIKResult:
        """Solve IK with leg stability constraints and return an interpolation.

        On whole-body Autolife chains, the seed is projected to a forward squat
        and every returned waypoint satisfies ``knee=2*ankle`` and
        ``abs(waist_pitch-ankle) < 60 degrees`` (limits are configurable).
        Waist pitch remains an independent cuRobo DOF.  Arm-only chains have no
        falling mode, so this behaves like :meth:`solve`.
        """
        cfg = config or self._config
        batch_seed = None
        if seed is not None:
            batch_seed = self._validate_seed(seed)[None, :]
        return self.solve_constrained_batch([target_pose], batch_seed, cfg)[0]

    def solve_constrained_batch(
        self,
        target_poses: Sequence[SE3Pose],
        seeds: np.ndarray | None = None,
        config: CuroboV2IKConfig | None = None,
    ) -> list[ConstrainedIKResult]:
        """Solve multiple targets with per-target Autolife stability rules.

        The knee coupling is part of cuRobo's batched robot model.  The
        waist-ankle bound is applied independently to candidates for each
        target, and every returned trajectory remains within both constraints.
        """
        cfg = config or self._config
        poses = self._validate_target_poses(target_poses)
        batch_seeds = self._validate_batch_seeds(seeds, len(poses))
        if not poses:
            return []
        stable_seeds = np.stack(
            [self._project_stable(seed, cfg) for seed in batch_seeds]
        )
        results = self._solve_batch_impl(
            poses,
            stable_seeds,
            cfg,
            constrained=self._stability_supported,
        )
        return [
            self._make_constrained_result(result, seed, cfg)
            for result, seed in zip(results, stable_seeds)
        ]

    def _solve_batch_impl(
        self,
        target_poses: list[SE3Pose],
        seeds: np.ndarray,
        config: CuroboV2IKConfig,
        *,
        constrained: bool,
    ) -> list[IKResult]:
        context = self._get_context(constrained, config)
        world_positions = np.stack(
            [np.asarray(pose.position, dtype=np.float64) for pose in target_poses]
        )
        world_rotations = np.stack(
            [np.asarray(pose.rotation, dtype=np.float64) for pose in target_poses]
        )
        local_positions = (world_positions - self._base_translation) @ (
            self._base_rotation
        )
        local_rotations = np.einsum(
            "ij,bjk->bik",
            self._base_rotation.T,
            world_rotations,
        )
        quaternion_xyzw = Rotation.from_matrix(local_rotations).as_quat()
        quaternion_wxyz = np.concatenate(
            [quaternion_xyzw[:, 3:4], quaternion_xyzw[:, :3]],
            axis=1,
        ).astype(np.float32)
        internal_seeds = np.stack(
            [self._public_to_internal(seed, context.joint_names) for seed in seeds]
        )

        torch = self._import_curobo_module("torch")
        tensor_kwargs = context.device_cfg.as_torch_dict()
        results: list[IKResult] = []
        execution_batch = config.max_batch_size
        for start in range(0, len(target_poses), execution_batch):
            end = min(start + execution_batch, len(target_poses))
            active_count = end - start
            position_chunk = local_positions[start:end]
            quaternion_chunk = quaternion_wxyz[start:end]
            seed_chunk = internal_seeds[start:end]

            if active_count < execution_batch:
                padding = execution_batch - active_count
                position_chunk = np.concatenate(
                    [position_chunk, np.repeat(position_chunk[-1:], padding, axis=0)]
                )
                quaternion_chunk = np.concatenate(
                    [
                        quaternion_chunk,
                        np.repeat(quaternion_chunk[-1:], padding, axis=0),
                    ]
                )
                seed_chunk = np.concatenate(
                    [seed_chunk, np.repeat(seed_chunk[-1:], padding, axis=0)]
                )

            target = context.pose_type(
                position=torch.as_tensor(
                    position_chunk.astype(np.float32), **tensor_kwargs
                ),
                quaternion=torch.as_tensor(quaternion_chunk, **tensor_kwargs),
            )
            goal = context.goal_tool_pose_type.from_poses(
                {self.ee_frame: target}, num_goalset=1
            )
            seed_tensor = torch.as_tensor(
                seed_chunk.astype(np.float32), **tensor_kwargs
            )
            current_state = None
            if config.use_current_state:
                current_state = context.joint_state_type.from_position(
                    seed_tensor,
                    joint_names=list(context.joint_names),
                )

            try:
                raw = context.solver.solve_pose(
                    goal_tool_poses=goal,
                    current_state=current_state,
                    seed_config=seed_tensor[:, None, :],
                    return_seeds=config.return_seeds,
                )
            except RuntimeError as exc:
                if "non-empty list of Tensors" not in str(exc):
                    raise
                results.extend(self._failed_result() for _ in range(active_count))
                continue

            candidates = self._batch_successful_candidates(
                raw,
                execution_batch,
                context,
                config,
            )
            for offset in range(active_count):
                index = start + offset
                results.append(
                    self._select_result(
                        candidates[offset],
                        seeds[index],
                        target_poses[index],
                        context,
                        config,
                        constrained=constrained,
                    )
                )
        return results

    def _select_result(
        self,
        candidates: list[np.ndarray],
        seed: np.ndarray,
        target_pose: SE3Pose,
        context: _CuroboContext,
        config: CuroboV2IKConfig,
        *,
        constrained: bool,
    ) -> IKResult:
        if not candidates:
            return self._failed_result()
        public_candidates = np.stack(
            [
                self._internal_to_public(row, context.joint_names, config)
                for row in candidates
            ]
        )
        if constrained and self._stability_supported:
            stable = np.asarray(
                [self._is_stable(row, config) for row in public_candidates],
                dtype=bool,
            )
            public_candidates = public_candidates[stable]
            if public_candidates.shape[0] == 0:
                return self._failed_result()
        best = int(np.argmin(np.linalg.norm(public_candidates - seed[None, :], axis=1)))
        solution = public_candidates[best]
        achieved = self.fk(solution)
        position_error = float(
            np.linalg.norm(achieved.position - np.asarray(target_pose.position))
        )
        rotation_error = achieved.rotation.T @ np.asarray(target_pose.rotation)
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(rotation_error).as_rotvec())
        )
        success = (
            position_error <= config.position_tolerance
            and orientation_error <= config.orientation_tolerance
        )
        return IKResult(
            status=IKStatus.SUCCESS if success else IKStatus.FAILED,
            joint_positions=solution,
            final_error=position_error + orientation_error,
            iterations=1,
            position_error=position_error,
            orientation_error=orientation_error,
        )

    def _make_constrained_result(
        self,
        result: IKResult,
        stable_seed: np.ndarray,
        config: CuroboV2IKConfig,
    ) -> ConstrainedIKResult:
        trajectory = None
        if result.joint_positions is not None:
            trajectory = np.linspace(
                stable_seed,
                result.joint_positions,
                num=config.trajectory_steps,
                dtype=np.float64,
            )
            if self._stability_supported:
                trajectory = np.stack(
                    [self._project_stable(row, config) for row in trajectory]
                )
        return ConstrainedIKResult(
            status=result.status,
            joint_positions=result.joint_positions,
            final_error=result.final_error,
            iterations=result.iterations,
            position_error=result.position_error,
            orientation_error=result.orientation_error,
            trajectory=trajectory,
        )

    def _get_context(
        self, constrained: bool, config: CuroboV2IKConfig
    ) -> _CuroboContext:
        key = (constrained, config)
        if key in self._contexts:
            return self._contexts[key]

        torch = self._import_curobo_module("torch")
        ik_module = self._import_curobo_module("curobo.inverse_kinematics")
        types_module = self._import_curobo_module("curobo.types")
        robot_module = self._import_curobo_module("curobo._src.types.robot")

        if not hasattr(torch, config.dtype):
            raise ValueError(f"Unknown torch dtype '{config.dtype}'")
        torch_dtype = getattr(torch, config.dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"'{config.dtype}' is not a torch dtype")
        device = "cuda:0" if config.tensor_device == "cuda" else config.tensor_device
        device_cfg = types_module.DeviceCfg(device=device, dtype=torch_dtype)

        urdf_path = self._chain_config.urdf_path
        if constrained and self._stability_supported:
            urdf_path = _stable_urdf_path(urdf_path, config)
        robot = robot_module.RobotCfg.from_basic(
            urdf_path,
            self.base_frame,
            [self.ee_frame],
            device_cfg=device_cfg,
            load_dynamics=False,
        )
        solver_config = ik_module.InverseKinematicsCfg.create(
            robot=robot,
            scene_model=None,
            collision_cache=None,
            self_collision_check=False,
            device_cfg=device_cfg,
            num_seeds=config.num_seeds,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            use_cuda_graph=config.use_cuda_graph,
            random_seed=config.random_seed,
            load_collision_spheres=False,
            max_batch_size=config.max_batch_size,
            override_optimizer_num_iters={
                "particle": config.particle_iters,
                "lbfgs": config.lbfgs_iters,
            },
        )
        solver = ik_module.InverseKinematics(solver_config)
        context = _CuroboContext(
            solver=solver,
            pose_type=types_module.Pose,
            goal_tool_pose_type=types_module.GoalToolPose,
            joint_state_type=types_module.JointState,
            device_cfg=device_cfg,
            joint_names=tuple(str(name) for name in solver.joint_names),
            constrained=constrained,
        )
        unknown = [
            name for name in context.joint_names if name not in self._name_to_public
        ]
        if unknown:
            raise ValueError(
                f"cuRobo exposed joints outside the requested chain: {unknown}"
            )
        self._contexts[key] = context
        return context

    def _batch_successful_candidates(
        self,
        raw: Any,
        batch_size: int,
        context: _CuroboContext,
        config: CuroboV2IKConfig,
    ) -> list[list[np.ndarray]]:
        dof = len(context.joint_names)
        solution = getattr(raw, "solution", None)
        if solution is not None:
            values = _as_numpy(solution, np.float64).reshape(batch_size, -1, dof)
        else:
            joint_state = getattr(raw, "js_solution", None)
            if joint_state is None or getattr(joint_state, "position", None) is None:
                return [[] for _ in range(batch_size)]
            full = _as_numpy(joint_state.position, np.float64)
            full = full.reshape(batch_size, -1, full.shape[-1])
            values = full[..., :dof]
        success = _as_numpy(raw.success, bool).reshape(batch_size, -1)
        candidates: list[list[np.ndarray]] = []
        for row in range(batch_size):
            columns = np.flatnonzero(success[row])
            columns = columns[columns < values.shape[1]][: config.return_seeds]
            candidates.append([values[row, column].copy() for column in columns])
        return candidates

    def _public_to_internal(
        self, public: np.ndarray, internal_names: tuple[str, ...]
    ) -> np.ndarray:
        return np.asarray(
            [public[self._name_to_public[name]] for name in internal_names],
            dtype=np.float64,
        )

    def _internal_to_public(
        self,
        internal: np.ndarray,
        internal_names: tuple[str, ...],
        config: CuroboV2IKConfig,
    ) -> np.ndarray:
        public = self._default_seed()
        for name, value in zip(internal_names, internal):
            public[self._name_to_public[name]] = float(value)
        if self._stability_supported and _KNEE not in internal_names:
            ankle = public[self._name_to_public[_ANKLE]]
            public[self._name_to_public[_KNEE]] = config.knee_multiplier * ankle
        return public

    def _project_stable(
        self, seed: np.ndarray, config: CuroboV2IKConfig
    ) -> np.ndarray:
        projected = np.asarray(seed, dtype=np.float64).copy()
        if not self._stability_supported:
            return projected
        ankle_index = self._name_to_public[_ANKLE]
        ankle = float(
            np.clip(
                projected[ankle_index],
                config.stability_ankle_min,
                config.stability_ankle_max,
            )
        )
        projected[ankle_index] = ankle
        projected[self._name_to_public[_KNEE]] = config.knee_multiplier * ankle
        waist_index = self._name_to_public[_WAIST_PITCH]
        tolerance = float(config.waist_ankle_tolerance)
        margin = min(1e-6, tolerance * 1e-6)
        difference = float(
            np.clip(
                projected[waist_index] - ankle,
                -tolerance + margin,
                tolerance - margin,
            )
        )
        projected[waist_index] = ankle + difference
        return projected

    def _is_stable(
        self, joint_positions: np.ndarray, config: CuroboV2IKConfig
    ) -> bool:
        ankle = joint_positions[self._name_to_public[_ANKLE]]
        waist = joint_positions[self._name_to_public[_WAIST_PITCH]]
        return bool(abs(waist - ankle) < config.waist_ankle_tolerance)

    def _validate_seed(self, seed: np.ndarray) -> np.ndarray:
        parsed = np.asarray(seed, dtype=np.float64)
        if parsed.shape != (self.num_joints,):
            raise ValueError(
                f"Expected seed shape ({self.num_joints},), got {parsed.shape}"
            )
        return parsed

    def _validate_batch_seeds(
        self,
        seeds: np.ndarray | None,
        batch_size: int,
    ) -> np.ndarray:
        if seeds is None:
            return np.repeat(
                self._default_seed()[None, :],
                batch_size,
                axis=0,
            )
        parsed = np.asarray(seeds, dtype=np.float64)
        expected = (batch_size, self.num_joints)
        if parsed.shape != expected:
            raise ValueError(f"Expected seeds shape {expected}, got {parsed.shape}")
        return parsed

    @staticmethod
    def _validate_target_poses(
        target_poses: Sequence[SE3Pose],
    ) -> list[SE3Pose]:
        poses = list(target_poses)
        invalid = [
            index
            for index, pose in enumerate(poses)
            if not isinstance(pose, SE3Pose)
        ]
        if invalid:
            raise TypeError(
                "target_poses entries must be SE3Pose; "
                f"invalid indices: {invalid}"
            )
        return poses

    def _default_seed(self) -> np.ndarray:
        neutral = self._pin.neutral(self._model)
        return np.asarray(
            [neutral[self._model.joints[jid].idx_q] for jid in self._joint_ids],
            dtype=np.float64,
        )

    def _to_full_q(self, joint_positions: np.ndarray) -> np.ndarray:
        q = self._pin.neutral(self._model)
        for value, joint_id in zip(joint_positions, self._joint_ids):
            q[self._model.joints[joint_id].idx_q] = float(value)
        return q

    @staticmethod
    def _failed_result() -> IKResult:
        return IKResult(
            status=IKStatus.FAILED,
            joint_positions=None,
            final_error=float("inf"),
            iterations=1,
            position_error=float("inf"),
            orientation_error=float("inf"),
        )

    @staticmethod
    def _import_curobo_module(name: str):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            missing = exc.name or name
            if (
                missing == "torch"
                or missing == "curobo"
                or missing.startswith("curobo.")
            ):
                raise CuroboV2UnavailableError(
                    "cuRoboV2 requires the optional CUDA dependencies. Install with "
                    "`pip install 'autolife_planning[curobo-v2]'`."
                ) from exc
            raise
