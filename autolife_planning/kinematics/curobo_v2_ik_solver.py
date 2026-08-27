"""CUDA-accelerated cuRoboV2 inverse kinematics.

The backend is optional and imported lazily.  Standard :meth:`solve` and
:meth:`solve_batch` use the original URDF, while their constrained variants use
an Autolife stability model that couples the knee to the ankle and bounds the
independent waist pitch relative to the ankle and limits waist yaw. IK optimizer
collision costs remain disabled; a separate lazy full-body cuRobo model provides
batched endpoint self- and point-cloud collision checks.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import tempfile
import time
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
_WAIST_YAW = "Joint_Waist_Yaw"
_STABILITY_JOINTS = {_ANKLE, _KNEE, _WAIST_PITCH, _WAIST_YAW}
_STABLE_URDF_SCHEMA = 4
_SPHERIZED_URDF_NAME = "autolife_spherized.urdf"
_COLLISION_CACHE_SCHEMA = 1


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
        f"{config.knee_multiplier}:"
        f"{config.stability_waist_yaw_min}:{config.stability_waist_yaw_max}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "autolife_planning_curobo_v2" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{source.stem}_stable.urdf"

    if not output.is_file():
        root = ET.fromstring(source.read_bytes())
        ankle = _find_joint(root, _ANKLE)
        knee = _find_joint(root, _KNEE)
        waist_yaw = _find_joint(root, _WAIST_YAW)

        ankle_min = float(config.stability_ankle_min)
        ankle_max = float(config.stability_ankle_max)
        _set_limit(ankle, ankle_min, ankle_max)
        _set_limit(
            knee,
            config.knee_multiplier * ankle_min,
            config.knee_multiplier * ankle_max,
        )
        _set_mimic(knee, _ANKLE, config.knee_multiplier)
        _set_limit(
            waist_yaw,
            config.stability_waist_yaw_min,
            config.stability_waist_yaw_max,
        )

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
    the fixed root-to-base transform is handled internally. Collision costs
    remain disabled in IK optimization; :meth:`is_in_collision_batch` uses a
    separate full-body sphere model for endpoint validation.
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

    def prepare_collision_checker(
        self,
        scene_points: np.ndarray,
        *,
        esdf_voxel_size: float = 0.04,
        warmup_batch_size: int = 32,
        joint_names: Sequence[str] | None = None,
        warmup_configuration: np.ndarray | None = None,
        point_radius: float = 0.0,
    ) -> dict[str, Any]:
        """Prepare cached collision artifacts and warm the GPU query kernel."""
        started = time.perf_counter()
        context = self._get_collision_context()
        model_seconds = time.perf_counter() - started
        esdf_started = time.perf_counter()
        esdf = self._get_esdf_context(scene_points, esdf_voxel_size, context)
        esdf_seconds = time.perf_counter() - esdf_started

        warmup_seconds = 0.0
        if warmup_configuration is not None:
            if warmup_batch_size < 1:
                raise ValueError("warmup_batch_size must be >= 1")
            seed = np.asarray(warmup_configuration, dtype=np.float64)
            if seed.ndim != 1:
                raise ValueError("warmup_configuration must be one-dimensional")
            warmup = np.repeat(seed[None, :], warmup_batch_size, axis=0)
            warmup_started = time.perf_counter()
            self.is_in_collision_batch(
                warmup,
                scene_points,
                joint_names=joint_names,
                point_radius=point_radius,
                configuration_batch_size=warmup_batch_size,
                scene_backend="esdf",
                esdf_voxel_size=esdf_voxel_size,
            )
            warmup_seconds = time.perf_counter() - warmup_started
        return {
            "model_seconds": model_seconds,
            "esdf_seconds": esdf_seconds,
            "warmup_seconds": warmup_seconds,
            "esdf_shape": esdf["shape"],
            "esdf_memory_bytes": esdf["memory_bytes"],
            "model_cache_hit": bool(context.get("disk_cache_hit", False)),
            "esdf_cache_hit": bool(esdf.get("disk_cache_hit", False)),
        }

    def are_spheres_in_collision_batch(
        self,
        centers: np.ndarray,
        scene_points: np.ndarray,
        *,
        radii: float | np.ndarray = 0.0,
        query_batch_size: int = 65_536,
        esdf_voxel_size: float = 0.04,
    ) -> np.ndarray:
        """Query arbitrary world-frame spheres against the cached GPU ESDF.

        This bypasses robot kinematics and is useful for testing sampled object
        geometry against the same static scene used by robot collision checks.
        One boolean is returned for every input sphere.
        """
        points = np.asarray(centers, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"centers must have shape (N, 3), got {points.shape}")
        scene = np.asarray(scene_points, dtype=np.float32)
        if scene.ndim != 2 or scene.shape[1] != 3:
            raise ValueError(
                f"scene_points must have shape (P, 3), got {scene.shape}"
            )
        if query_batch_size < 1 or esdf_voxel_size <= 0.0:
            raise ValueError("query batch size and ESDF voxel size must be > 0")
        radius_values = np.asarray(radii, dtype=np.float32)
        if radius_values.ndim == 0:
            radius_values = np.full(len(points), float(radius_values), np.float32)
        elif radius_values.shape != (len(points),):
            raise ValueError("radii must be scalar or have shape (N,)")
        if np.any(~np.isfinite(radius_values)) or np.any(radius_values < 0.0):
            raise ValueError("radii must be finite and >= 0")
        if len(points) == 0 or len(scene) == 0:
            return np.zeros(len(points), dtype=bool)

        context = self._get_collision_context()
        esdf = self._get_esdf_context(scene, esdf_voxel_size, context)
        output = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), query_batch_size):
            stop = min(start + query_batch_size, len(points))
            spheres = np.column_stack(
                (points[start:stop], radius_values[start:stop])
            ).astype(np.float32, copy=False)
            query = (
                context["device_cfg"]
                .to_device(spheres)
                .reshape(stop - start, 1, 1, 4)
                .contiguous()
            )
            buffer = esdf["buffer_type"].from_shape(
                query.shape, context["device_cfg"]
            )
            cost = esdf["scene"].get_sphere_distance_raw(
                query,
                buffer,
                esdf["weight"],
                esdf["activation_distance"],
            )
            output[start:stop] = _as_numpy(
                cost.reshape(stop - start, -1).amax(dim=1) > 0.0,
                bool,
            )
        return output

    def is_in_collision_batch(
        self,
        joint_positions: np.ndarray,
        scene_points: np.ndarray,
        *,
        joint_names: Sequence[str] | None = None,
        ignore_scene_links: Sequence[str] | None = None,
        check_self_collision: bool = True,
        point_radius: float = 0.0,
        configuration_batch_size: int = 32,
        point_batch_size: int = 4096,
        scene_backend: str = "esdf",
        esdf_voxel_size: float = 0.04,
    ) -> np.ndarray:
        """Check configurations against self-collision and a point cloud on GPU.

        Args:
            joint_positions: Configurations with shape ``(N, D)``.
            scene_points: Obstacle centers in the URDF root/world frame, with
                shape ``(P, 3)``. When virtual mobile-base joints are supplied,
                their X/Y/yaw values place that root frame in the same world.
            joint_names: Names corresponding to the configuration columns.
                Defaults to this IK solver's :attr:`joint_names`.
            ignore_scene_links: Robot links excluded only from scene collision.
                They remain active in robot self-collision. This is useful for
                intentional contact, such as a gripper touching an object.
            check_self_collision: Whether to include robot self-collision in
                the returned mask. Disable this when composing a second,
                dynamic scene query with a static query that already checked
                self-collision.
            point_radius: Radius assigned to every obstacle point.
            configuration_batch_size: Maximum configurations evaluated together.
            point_batch_size: Maximum obstacle points evaluated together by
                the direct sphere-point fallback.
            scene_backend: ``"esdf"`` for cuRobo's GPU voxel kernel or
                ``"points"`` for direct sphere-point comparisons.
            esdf_voxel_size: Voxel resolution used by the ESDF backend.

        Returns:
            Boolean array of shape ``(N,)`` where ``True`` means collision.
        """
        if point_radius < 0.0:
            raise ValueError("point_radius must be >= 0")
        if configuration_batch_size < 1 or point_batch_size < 1:
            raise ValueError("collision batch sizes must be >= 1")
        scene_backend = scene_backend.lower()
        if scene_backend not in {"esdf", "points"}:
            raise ValueError("scene_backend must be 'esdf' or 'points'")
        if esdf_voxel_size <= 0.0:
            raise ValueError("esdf_voxel_size must be > 0")
        configurations = np.asarray(joint_positions, dtype=np.float64)
        names = tuple(self.joint_names if joint_names is None else joint_names)
        if configurations.ndim != 2 or configurations.shape[1] != len(names):
            raise ValueError(
                f"joint_positions must have shape (N, {len(names)}), "
                f"got {configurations.shape}"
            )
        if len(set(names)) != len(names):
            raise ValueError("joint_names must not contain duplicates")
        points = np.asarray(scene_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"scene_points must have shape (P, 3), got {points.shape}")
        if configurations.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        context = self._get_collision_context()
        torch = self._import_curobo_module("torch")
        input_indices = {name: index for index, name in enumerate(names)}
        missing = [name for name in context["joint_names"] if name not in input_indices]
        if missing:
            raise ValueError(f"joint_positions are missing collision-model joints: {missing}")
        base_names = ("Joint_Virtual_X", "Joint_Virtual_Y", "Joint_Virtual_Theta")
        has_mobile_base = all(name in input_indices for name in base_names)
        point_tensor = None
        esdf_context = None
        ignored_sphere_indices: list[int] = []
        if ignore_scene_links:
            kinematics_config = context["checker"].kinematics.config.kinematics_config
            unknown_links = [
                str(link)
                for link in ignore_scene_links
                if str(link) not in kinematics_config.link_name_to_idx_map
            ]
            if unknown_links:
                raise ValueError(
                    "ignore_scene_links contains links outside the collision model: "
                    f"{unknown_links}"
                )
            for link in dict.fromkeys(str(value) for value in ignore_scene_links):
                ignored_sphere_indices.extend(
                    int(value)
                    for value in kinematics_config.get_sphere_index_from_link_name(
                        link
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
        if scene_backend == "esdf" and points.shape[0] > 0:
            esdf_context = self._get_esdf_context(points, esdf_voxel_size, context)
        elif points.shape[0] > 0:
            point_tensor = context["device_cfg"].to_device(points)
        output = np.zeros(configurations.shape[0], dtype=bool)

        for config_start in range(0, configurations.shape[0], configuration_batch_size):
            config_stop = min(
                config_start + configuration_batch_size, configurations.shape[0]
            )
            config_chunk = configurations[config_start:config_stop]
            internal = config_chunk[
                :, [input_indices[name] for name in context["joint_names"]]
            ]
            q = context["device_cfg"].to_device(internal).unsqueeze(1).contiguous()
            checker = context["checker"]
            checker.setup_batch_tensors(config_chunk.shape[0], 1)
            state = checker.get_kinematics(q)
            spheres = state.robot_spheres.reshape(config_chunk.shape[0], -1, 4)
            if check_self_collision:
                self_cost = checker.get_self_collision_distance(
                    spheres.unsqueeze(1)
                )
                in_collision = (
                    self_cost.reshape(config_chunk.shape[0], -1).amax(dim=1)
                    > 0.0
                )
            else:
                in_collision = torch.zeros(
                    config_chunk.shape[0],
                    dtype=torch.bool,
                    device=spheres.device,
                )

            scene_spheres = spheres
            if ignored_sphere_indices:
                scene_spheres = spheres.clone()
                # A large negative radius makes only these spheres inert for
                # scene queries. The unmodified spheres above already went
                # through the full self-collision check.
                scene_spheres[..., ignored_sphere_indices, 3] = -1.0e6

            if has_mobile_base:
                x = context["device_cfg"].to_device(
                    config_chunk[:, input_indices[base_names[0]]]
                )
                y = context["device_cfg"].to_device(
                    config_chunk[:, input_indices[base_names[1]]]
                )
                yaw = context["device_cfg"].to_device(
                    config_chunk[:, input_indices[base_names[2]]]
                )
                # cuRobo's fixed-base model reports sphere centers in
                # ``self.base_frame`` (Link_Ground_Vehicle), whereas the
                # virtual X/Y/yaw coordinates place the original URDF root
                # (Link_Zero_Point) in the world. Restore the fixed root-to-
                # base transform before applying the sampled mobile pose.
                # Omitting this transform rotates the complete sphere model
                # onto the ground and shifts it away from the rendered robot.
                root_from_base_rotation = context["device_cfg"].to_device(
                    self._base_rotation
                )
                root_from_base_translation = context["device_cfg"].to_device(
                    self._base_translation
                )
                root_centers = (
                    scene_spheres[..., :3]
                    @ root_from_base_rotation.transpose(0, 1)
                    + root_from_base_translation
                )
                root_x = root_centers[..., 0]
                root_y = root_centers[..., 1]
                cosine = torch.cos(yaw)[:, None]
                sine = torch.sin(yaw)[:, None]
                scene_spheres[..., 0] = (
                    x[:, None] + cosine * root_x - sine * root_y
                )
                scene_spheres[..., 1] = (
                    y[:, None] + sine * root_x + cosine * root_y
                )
                scene_spheres[..., 2] = root_centers[..., 2]

            if esdf_context is not None:
                query_spheres = scene_spheres.unsqueeze(1).contiguous()
                query_spheres[..., 3] += float(point_radius)
                buffer = esdf_context["buffer_type"].from_shape(
                    query_spheres.shape, context["device_cfg"]
                )
                scene_cost = esdf_context["scene"].get_sphere_distance_raw(
                    query_spheres,
                    buffer,
                    esdf_context["weight"],
                    esdf_context["activation_distance"],
                )
                in_collision |= (
                    scene_cost.reshape(config_chunk.shape[0], -1).amax(dim=1) > 0.0
                )
            elif point_tensor is not None:
                for point_start in range(0, points.shape[0], point_batch_size):
                    active = ~in_collision
                    if not bool(torch.any(active)):
                        break
                    point_chunk = point_tensor[point_start : point_start + point_batch_size]
                    active_spheres = scene_spheres[active]
                    deltas = (
                        point_chunk[None, :, None, :]
                        - active_spheres[:, None, :, :3]
                    )
                    signed = (
                        active_spheres[:, None, :, 3]
                        + float(point_radius)
                        - torch.linalg.norm(deltas, dim=-1)
                    )
                    active_collision = signed.amax(dim=(1, 2)) >= 0.0
                    in_collision[active] |= active_collision
            output[config_start:config_stop] = _as_numpy(in_collision, bool)
        return output

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
        """Solve IK with leg stability constraints and optional interpolation.

        On whole-body Autolife chains, the seed is projected to a forward squat
        and every returned waypoint satisfies ``knee=2*ankle`` and
        ``-10 <= waist_pitch-ankle <= 60 degrees``.  Waist yaw is limited to
        ``[-75, 75] degrees`` (all limits are configurable).
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
        waist-ankle bound is a differentiable constraint in cuRobo's optimizer
        and feasibility rollout.  When requested, every returned trajectory
        remains within both constraints.
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
        if config.return_trajectory and result.joint_positions is not None:
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
        constraint_module = None
        extra_solver_config: dict[str, Any] = {}
        if constrained and self._stability_supported:
            constraint_module = importlib.import_module(
                "autolife_planning.kinematics.curobo_v2_constraints"
            )
            extra_solver_config["cost_manager_config_instance_type"] = (
                constraint_module.AutolifeRobotCostManagerCfg
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
            **extra_solver_config,
        )
        if constraint_module is not None:
            constraint_module.add_waist_ankle_constraint(
                solver_config,
                lower_bound=config.waist_ankle_min,
                upper_bound=config.waist_ankle_max,
                weight=config.waist_ankle_constraint_weight,
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

    def _get_collision_context(self) -> dict[str, Any]:
        """Create the lazily cached full-body cuRobo collision model."""
        cached = getattr(self, "_collision_context", None)
        if cached is not None:
            return cached

        torch = self._import_curobo_module("torch")
        types_module = self._import_curobo_module("curobo.types")
        robot_module = self._import_curobo_module("curobo._src.types.robot")
        loader_module = self._import_curobo_module(
            "curobo._src.robot.loader.kinematics_loader_cfg"
        )
        builder_module = self._import_curobo_module(
            "curobo._src.robot.builder.builder_robot"
        )
        collision_cfg_module = self._import_curobo_module(
            "curobo._src.collision.collision_robot_scene_cfg"
        )
        collision_module = self._import_curobo_module(
            "curobo._src.collision.collision_robot_scene"
        )

        device = "cuda:0" if self._config.tensor_device == "cuda" else self._config.tensor_device
        torch_dtype = getattr(torch, self._config.dtype)
        device_cfg = types_module.DeviceCfg(device=device, dtype=torch_dtype)
        sphere_urdf = Path(self._chain_config.urdf_path).with_name(_SPHERIZED_URDF_NAME)
        if not sphere_urdf.is_file():
            raise FileNotFoundError(f"cuRobo collision URDF not found: {sphere_urdf}")

        root = ET.parse(sphere_urdf).getroot()
        collision_spheres: dict[str, list[dict[str, Any]]] = {}
        for link in root.findall("link"):
            spheres: list[dict[str, Any]] = []
            for collision in link.findall("collision"):
                sphere = collision.find("geometry/sphere")
                if sphere is None:
                    continue
                origin = collision.find("origin")
                center = [0.0, 0.0, 0.0]
                if origin is not None:
                    center = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
                spheres.append(
                    {"center": center, "radius": float(sphere.attrib["radius"])}
                )
            if spheres:
                collision_spheres[link.attrib["name"]] = spheres
        if not collision_spheres:
            raise ValueError(f"No collision spheres found in {sphere_urdf}")

        # Generate cuRobo's ignore matrix once for this sphere model. Besides
        # adjacent links, this suppresses intentional overlaps in the neutral
        # CAD decomposition (camera mounts, shoulders, and gripper fingers).
        model_digest = hashlib.sha256()
        model_digest.update(str(_COLLISION_CACHE_SCHEMA).encode())
        model_digest.update(Path(self._chain_config.urdf_path).read_bytes())
        model_digest.update(sphere_urdf.read_bytes())
        model_cache_dir = Path.home() / ".cache" / "autolife_planning" / "curobo_v2"
        model_cache_path = model_cache_dir / f"self_collision_{model_digest.hexdigest()}.json"
        model_cache_hit = model_cache_path.is_file()
        if model_cache_hit:
            self_collision_ignore = json.loads(model_cache_path.read_text())
        else:
            yourdfpy_logger = logging.getLogger("yourdfpy.urdf")
            previous_level = yourdfpy_logger.level
            try:
                yourdfpy_logger.setLevel(logging.ERROR)
                builder = builder_module.RobotBuilder(
                    self._chain_config.urdf_path,
                    asset_path=str(Path(self._chain_config.urdf_path).parent),
                    tool_frames=[self.ee_frame],
                    device_cfg=device_cfg,
                )
                builder._collision_spheres = collision_spheres
                self_collision_ignore = builder.compute_collision_matrix(
                    prune_collisions=False
                )
            finally:
                yourdfpy_logger.setLevel(previous_level)
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = model_cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self_collision_ignore, sort_keys=True))
            os.replace(temporary, model_cache_path)

        # Use the regular URDF for kinematics and the pre-generated sphere
        # decomposition for geometry. The collision model is rooted at the
        # robot base link; is_in_collision_batch() maps its sphere centers
        # through the fixed URDF root-to-base transform and then through each
        # candidate virtual-base pose before querying world-frame obstacles.
        loader_cfg = loader_module.KinematicsLoaderCfg(
            base_link=self.base_frame,
            tool_frames=[self.ee_frame],
            urdf_path=self._chain_config.urdf_path,
            collision_link_names=list(collision_spheres),
            collision_spheres=collision_spheres,
            self_collision_buffer={},
            self_collision_ignore=self_collision_ignore,
            device_cfg=device_cfg,
        )
        robot = robot_module.RobotCfg.create(
            {
                "kinematics": {
                    key: value
                    for key, value in vars(loader_cfg).items()
                    if key not in {"device_cfg", "load_collision_spheres", "num_envs"}
                }
            },
            device_cfg=device_cfg,
        )
        checker_cfg = collision_cfg_module.RobotSceneCollisionCfg.load_from_config(
            robot_config=robot,
            device_cfg=device_cfg,
            scene_model=None,
            self_collision_activation_distance=0.0,
        )
        checker = collision_module.RobotSceneCollision(checker_cfg)
        context = {
            "checker": checker,
            "device_cfg": device_cfg,
            "joint_names": tuple(str(name) for name in checker.kinematics.joint_names),
            "disk_cache_hit": model_cache_hit,
        }
        self._collision_context = context
        return context

    def _get_esdf_context(
        self,
        points: np.ndarray,
        voxel_size: float,
        collision_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build and cache a GPU ESDF scene for a static point cloud."""
        contiguous = np.ascontiguousarray(points, dtype=np.float32)
        digest = hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
        key = (digest, float(voxel_size))
        cache = getattr(self, "_esdf_contexts", None)
        if cache is None:
            cache = {}
            self._esdf_contexts = cache
        if key in cache:
            return cache[key]

        from scipy.ndimage import distance_transform_edt

        # Include enough free-space padding for interpolation around robot
        # spheres near the observed cloud boundary.
        margin = max(0.5, 4.0 * voxel_size)
        lower = contiguous.min(axis=0).astype(np.float64) - margin
        upper = contiguous.max(axis=0).astype(np.float64) + margin
        shape = np.ceil((upper - lower) / voxel_size).astype(np.int64) + 1
        center = 0.5 * (lower + upper)
        dims = shape.astype(np.float64) * voxel_size

        esdf_cache_dir = Path.home() / ".cache" / "autolife_planning" / "curobo_v2"
        esdf_cache_path = esdf_cache_dir / f"esdf_{digest}_{voxel_size:.6f}.npz"
        esdf_cache_hit = esdf_cache_path.is_file()
        if esdf_cache_hit:
            with np.load(esdf_cache_path) as cached_esdf:
                signed_distance = np.asarray(cached_esdf["signed_distance"], dtype=np.float16)
                lower = np.asarray(cached_esdf["lower"], dtype=np.float64)
                shape = np.asarray(signed_distance.shape, dtype=np.int64)
                center = 0.5 * (lower + lower + (shape - 1) * voxel_size)
                dims = shape.astype(np.float64) * voxel_size
        else:
            occupancy = np.zeros(tuple(int(value) for value in shape), dtype=bool)
            indices = np.rint(
                (contiguous.astype(np.float64) - lower) / voxel_size
            ).astype(np.int64)
            indices = np.clip(indices, 0, shape - 1)
            occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = True
            outside = distance_transform_edt(~occupancy).astype(np.float32) * voxel_size
            inside = distance_transform_edt(occupancy).astype(np.float32) * voxel_size
            signed_distance = (outside - inside).astype(np.float16)
            esdf_cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = esdf_cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                signed_distance=signed_distance,
                lower=lower,
            )
            os.replace(temporary, esdf_cache_path)

        torch = self._import_curobo_module("torch")
        geom_module = self._import_curobo_module("curobo._src.geom.types")
        scene_module = self._import_curobo_module(
            "curobo._src.geom.collision.collision_scene"
        )
        buffer_module = self._import_curobo_module(
            "curobo._src.geom.collision.buffer_collision"
        )
        device_cfg = collision_context["device_cfg"]
        feature = torch.as_tensor(
            signed_distance,
            device=device_cfg.device,
            dtype=torch.float16,
        ).contiguous()
        voxel = geom_module.VoxelGrid(
            name="autolife_rls_esdf",
            pose=[
                float(center[0]),
                float(center[1]),
                float(center[2]),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dims=[float(value) for value in dims],
            voxel_size=float(voxel_size),
            feature_tensor=feature,
            feature_dtype=torch.float16,
        )
        scene_cfg = geom_module.SceneCfg(voxel=[voxel])
        scene = scene_module.create_scene_collision(
            scene_module.SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=scene_cfg,
                cache={"voxel": 1},
                max_distance=max(1.0, margin),
            )
        )
        result = {
            "scene": scene,
            "buffer_type": buffer_module.CollisionBuffer,
            "weight": device_cfg.to_device([1.0]),
            "activation_distance": device_cfg.to_device([0.0]),
            "shape": tuple(int(value) for value in shape),
            "memory_bytes": int(feature.numel() * feature.element_size()),
            "disk_cache_hit": esdf_cache_hit,
        }
        cache[key] = result
        return result

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
        difference = float(
            np.clip(
                projected[waist_index] - ankle,
                config.waist_ankle_min,
                config.waist_ankle_max,
            )
        )
        projected[waist_index] = ankle + difference
        waist_yaw_index = self._name_to_public[_WAIST_YAW]
        projected[waist_yaw_index] = np.clip(
            projected[waist_yaw_index],
            config.stability_waist_yaw_min,
            config.stability_waist_yaw_max,
        )
        return projected

    def _is_stable(
        self, joint_positions: np.ndarray, config: CuroboV2IKConfig
    ) -> bool:
        ankle = joint_positions[self._name_to_public[_ANKLE]]
        waist = joint_positions[self._name_to_public[_WAIST_PITCH]]
        waist_yaw = joint_positions[self._name_to_public[_WAIST_YAW]]
        difference = waist - ankle
        return bool(
            config.waist_ankle_min <= difference <= config.waist_ankle_max
            and config.stability_waist_yaw_min
            <= waist_yaw
            <= config.stability_waist_yaw_max
        )

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
