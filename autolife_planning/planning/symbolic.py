"""Symbolic forward kinematics and CasADi compilation.

:class:`SymbolicContext` turns the planner's active joint vector into
CasADi expressions for link poses, built directly from the URDF: the
planar base ``(x, y, theta)`` followed by each joint's fixed origin and
its revolute / prismatic motion.  :func:`compile_function` generates C
for a CasADi function, compiles it to a shared library and caches it by
content hash; :class:`~autolife_planning.planning.constraints.Constraint`
and :class:`~autolife_planning.planning.costs.Cost` hand the result to
the C++ planner.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from autolife_planning.autolife import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    autolife_robot_config,
)


@dataclass(frozen=True)
class _Joint:
    name: str
    type: str
    parent: str
    origin: np.ndarray  # 4x4 transform, parent link -> joint frame
    axis: np.ndarray  # unit axis in the joint frame


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _parse_joints(urdf_path: str) -> dict[str, _Joint]:
    """URDF joints keyed by their child link."""
    joints = {}
    for j in ET.parse(urdf_path).getroot().iter("joint"):
        origin = j.find("origin")
        xyz = rpy = "0 0 0"
        if origin is not None:
            xyz = origin.get("xyz", xyz)
            rpy = origin.get("rpy", rpy)
        T = np.eye(4)
        T[:3, :3] = _rpy_matrix(*map(float, rpy.split()))
        T[:3, 3] = [float(v) for v in xyz.split()]
        axis = j.find("axis")
        a = np.array(
            [
                float(v)
                for v in (axis.get("xyz") if axis is not None else "1 0 0").split()
            ]
        )
        if j.get("type") != "fixed":
            a /= np.linalg.norm(a)
        child = j.find("child").get("link")
        joints[child] = _Joint(
            j.get("name"), j.get("type"), j.find("parent").get("link"), T, a
        )
    return joints


def _joint_motion(joint: _Joint, q) -> ca.SX:
    """4x4 transform produced by moving ``joint`` to position ``q``."""
    T = ca.SX.eye(4)
    if joint.type in ("revolute", "continuous"):
        # Rodrigues: R = I + sin(q) K + (1 - cos(q)) K^2
        x, y, z = joint.axis
        K = ca.SX(np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]))
        T[:3, :3] = ca.SX.eye(3) + ca.sin(q) * K + (1 - ca.cos(q)) * (K @ K)
    elif joint.type == "prismatic":
        T[:3, 3] = ca.SX(joint.axis) * q
    return T


class SymbolicContext:
    """CasADi-friendly view of the planner's active subgroup.

    Holds the CasADi symbolic joint vector ``q`` matching the subgroup's
    active dimension.  Joints outside the subgroup are frozen at
    ``base_config``, the same 24-DOF body vector
    (``[x, y, theta, j0..j20]``) the planner uses.
    """

    def __init__(
        self,
        subgroup: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        if base_config is None:
            base_config = HOME_JOINTS
        self.base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self.base_config.shape != HOME_JOINTS.shape:
            raise ValueError(
                f"base_config must have shape {HOME_JOINTS.shape}, "
                f"got {self.base_config.shape}"
            )

        self.subgroup_name = subgroup
        self._names = list(autolife_robot_config.joint_names)
        if subgroup == "autolife":
            self.active_names = list(self._names)
        else:
            sg = PLANNING_SUBGROUPS.get(subgroup)
            if sg is None:
                raise ValueError(f"Unknown subgroup: {subgroup!r}")
            self.active_names = list(sg["joints"])
        self.active_indices = [self._names.index(j) for j in self.active_names]

        self.q = ca.SX.sym("q", len(self.active_indices))
        self._joints = _parse_joints(autolife_robot_config.urdf_path)
        self._poses: dict[str, ca.SX] = {}
        self._pose_fns: dict[str, ca.Function] = {}

    def _pose(self, link_name: str, q_active: ca.SX) -> ca.SX:
        full = [ca.SX(float(v)) for v in self.base_config]
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active[i]
        value = dict(zip(self._names, full))

        chain = []
        link = link_name
        while link in self._joints:
            chain.append(self._joints[link])
            link = self._joints[link].parent
        if not chain and link_name not in {j.parent for j in self._joints.values()}:
            raise ValueError(f"Unknown link: {link_name!r}")

        # Planar base (x, y, theta) at the URDF root.
        x, y, theta = full[0], full[1], full[2]
        T = ca.SX.eye(4)
        T[0, 0], T[0, 1], T[1, 0], T[1, 1] = (
            ca.cos(theta),
            -ca.sin(theta),
            ca.sin(theta),
            ca.cos(theta),
        )
        T[0, 3], T[1, 3] = x, y
        for joint in reversed(chain):
            T = T @ ca.SX(joint.origin)
            if joint.type != "fixed":
                # Movable URDF joints outside the 24-DOF body stay at zero.
                T = T @ _joint_motion(joint, value.get(joint.name, 0.0))
        return T

    def link_pose(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 4x4 world pose of a URDF link.

        Uses ``self.q`` unless another active-dimension symbol is given.
        """
        if q_active is not None and q_active is not self.q:
            return self._pose(link_name, q_active)
        if link_name not in self._poses:
            self._poses[link_name] = self._pose(link_name, self.q)
        return self._poses[link_name]

    def link_translation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3-vector: link position in world frame."""
        return self.link_pose(link_name, q_active)[:3, 3]

    def link_rotation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3x3 rotation matrix of the link."""
        return self.link_pose(link_name, q_active)[:3, :3]

    def evaluate_link_pose(
        self, link_name: str, q_active_numeric: np.ndarray
    ) -> np.ndarray:
        """Compute a NUMERIC 4x4 link pose (handy for building targets)."""
        if link_name not in self._pose_fns:
            self._pose_fns[link_name] = ca.Function(
                "fk", [self.q], [self.link_pose(link_name)]
            )
        return np.asarray(self._pose_fns[link_name](q_active_numeric), dtype=np.float64)

    def project(
        self,
        q_init: np.ndarray,
        residual: ca.SX,
        tol: float = 1e-8,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Project a joint configuration onto the manifold ``residual(q) = 0``.

        Runs damped Gauss-Newton on the CasADi Jacobian — the same
        iteration OMPL's ``ProjectedStateSpace`` runs internally — so
        the returned config will pass the planner's tolerance check
        and can be used directly as a start or goal state.
        """
        res_fn = ca.Function("proj_res", [self.q], [ca.reshape(residual, -1, 1)])
        jac_fn = ca.Function("proj_jac", [self.q], [ca.jacobian(residual, self.q)])
        q = np.asarray(q_init, dtype=np.float64).copy()
        for _ in range(max_iters):
            r = np.asarray(res_fn(q)).flatten()
            if np.linalg.norm(r) < tol:
                return q
            J = np.asarray(jac_fn(q))
            JJt = J @ J.T + 1e-10 * np.eye(J.shape[0])
            q -= J.T @ np.linalg.solve(JJt, r)
        raise RuntimeError(
            f"SymbolicContext.project failed to converge: "
            f"|residual|={np.linalg.norm(r):.2e} after {max_iters} iters"
        )


def _cache_root() -> Path:
    """``$AUTOLIFE_CASADI_CACHE_DIR``, else ``~/.cache/autolife_planning``."""
    override = os.environ.get("AUTOLIFE_CASADI_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "autolife_planning").resolve()


def compile_function(f: ca.Function, kind: str) -> Path:
    """Compile a CasADi function to a shared library, cached by content.

    The C code CasADi generates is compiled with ``$CXX`` (default
    ``c++``) into ``<cache>/<kind>/<sha256>/<kind>.so``; a cache hit
    costs one ``stat``.
    """
    raw = f.serialize()
    sha = hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()
    cache_dir = _cache_root() / kind / sha[:2] / sha[2:]
    so_path = cache_dir / f"{kind}.so"
    if so_path.exists():
        return so_path

    sys.stderr.write(f"[autolife] compiling {kind} {sha[:8]}... ")
    sys.stderr.flush()
    t0 = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cg = ca.CodeGenerator(f"{kind}.c")
    cg.add(f)
    cg.generate(f"{cache_dir}{os.sep}")
    # Compile under a per-process name, then rename: a concurrent reader
    # never sees a half-written library.
    tmp = cache_dir / f"{kind}.{os.getpid()}.so"
    cxx = os.environ.get("CXX", "c++")
    subprocess.run(
        [cxx, "-O3", "-shared", "-fPIC", str(cache_dir / f"{kind}.c"), "-o", str(tmp)],
        check=True,
    )
    os.replace(tmp, so_path)
    sys.stderr.write(f"done ({(time.perf_counter() - t0) * 1000:.0f} ms)\n")
    sys.stderr.flush()
    return so_path


__all__ = ["SymbolicContext", "compile_function"]
