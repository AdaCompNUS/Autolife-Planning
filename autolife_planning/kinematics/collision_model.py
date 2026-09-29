"""Pinocchio collision model builder.

Builds a GeometryModel with self-collision pairs (from URDF + SRDF) and
supports adding pointcloud obstacles as sphere primitives.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

pin = importlib.import_module("pinocchio")


@dataclass
class CollisionContext:
    """Pinocchio model with collision geometry."""

    model: Any  # pin.Model
    collision_model: Any  # pin.GeometryModel
    data: Any  # pin.Data
    collision_data: Any  # pin.GeometryData


def build_collision_model(
    urdf_path: str,
    srdf_path: str | None = None,
    mesh_dir: str | None = None,
    min_distance: float = 0.01,
) -> CollisionContext:
    """Build a Pinocchio model with collision geometry from URDF + SRDF.

    After SRDF filtering, pairs that are already closer than *min_distance*
    at the neutral configuration are removed.  These are typically adjacent
    links whose meshes overlap and would make the QP infeasible.

    Input:
        urdf_path: Path to the URDF file.
        srdf_path: Path to SRDF for collision pair filtering.
            If None, auto-resolves from same directory as URDF.
        mesh_dir: Directory for resolving ``package://`` mesh URIs.
            Defaults to the URDF's parent directory.
        min_distance: Pairs closer than this at neutral are pruned (meters).
    Output:
        CollisionContext with model, collision model, and associated data.
    """
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    if mesh_dir is None:
        mesh_dir = urdf_dir

    if srdf_path is None:
        base = os.path.splitext(os.path.basename(urdf_path))[0]
        candidate = os.path.join(urdf_dir, f"{base}.srdf")
        if os.path.exists(candidate):
            srdf_path = candidate

    model, collision_model, _ = pin.buildModelsFromUrdf(urdf_path, mesh_dir)

    collision_model.addAllCollisionPairs()
    if srdf_path is not None:
        pin.removeCollisionPairs(model, collision_model, srdf_path)

    # Prune pairs already in contact at neutral — these are adjacent-link
    # overlaps that would make any collision barrier infeasible.
    data = model.createData()
    collision_data = pin.GeometryData(collision_model)
    q_neutral = pin.neutral(model)
    pin.computeDistances(model, data, collision_model, collision_data, q_neutral)

    to_remove = []
    for i, cr in enumerate(collision_data.distanceResults):
        if cr.min_distance < min_distance:
            to_remove.append(collision_model.collisionPairs[i])

    for pair in to_remove:
        collision_model.removeCollisionPair(pair)

    # Rebuild data after pruning
    collision_data = pin.GeometryData(collision_model)

    return CollisionContext(
        model=model,
        collision_model=collision_model,
        data=data,
        collision_data=collision_data,
    )


def add_pointcloud_obstacles(
    context: CollisionContext,
    points: np.ndarray,
    radius: float = 0.02,
    voxel_size: float | None = None,
) -> int:
    """Add pointcloud obstacles as spheres to the collision model.

    Each point becomes a sphere attached to the universe joint.
    Collision pairs are added between every robot geometry and every
    new obstacle sphere so that ``SelfCollisionBarrier`` checks them.

    Input:
        context: Collision context to modify *in place*.
        points: (N, 3) array of obstacle positions in world frame.
        radius: Sphere radius in meters.
        voxel_size: If set, downsample the pointcloud to a voxel grid first.
    Output:
        Number of obstacle spheres added.
    """
    # Pinocchio >= 3 ships its collision library as ``coal`` (the renamed
    # hpp-fcl); Pinocchio 4 no longer pulls in the ``hppfcl`` compatibility
    # shim, so prefer coal and keep hppfcl as the fallback for older stacks.
    try:
        import coal as fcl
    except ImportError:  # pragma: no cover - legacy hpp-fcl stacks
        import hppfcl as fcl

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points array, got {points.shape}")

    if voxel_size is not None and voxel_size > 0:
        quantized = np.round(points / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        points = quantized[unique_idx].astype(np.float64) * voxel_size

    n_robot_geoms = context.collision_model.ngeoms
    n_added = 0

    for i, pt in enumerate(points):
        sphere = fcl.Sphere(radius)
        placement = pin.SE3(np.eye(3), pt)
        # GeometryObject's constructor argument order differs across
        # Pinocchio majors:
        #   >= 3 (incl. 4): (name, parent_joint, placement, geometry)
        #   3.x deprecated: (name, parent_joint, geometry, placement)
        #   2.x:            (name, parent_frame, parent_joint, geometry, placement)
        # Try the current form first, then fall back to the older ones.
        name = f"obstacle_{i}"
        parent_joint = 0  # universe
        parent_frame = 0  # universe frame
        geom = None
        for args in (
            (name, parent_joint, placement, sphere),
            (name, parent_joint, sphere, placement),
            (name, parent_frame, parent_joint, sphere, placement),
        ):
            try:
                geom = pin.GeometryObject(*args)
                break
            except Exception:
                continue
        if geom is None:
            raise RuntimeError(
                "Unsupported pinocchio.GeometryObject constructor signature; "
                f"pinocchio {getattr(pin, '__version__', '?')}"
            )
        obs_id = context.collision_model.addGeometryObject(geom)

        for robot_id in range(n_robot_geoms):
            context.collision_model.addCollisionPair(
                pin.CollisionPair(robot_id, obs_id)
            )
        n_added += 1

    context.collision_data = pin.GeometryData(context.collision_model)
    return n_added
