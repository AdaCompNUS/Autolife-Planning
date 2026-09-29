"""Decide which self-collision pairs the SRDF may disable, from data.

Every pair of links that carries spheres in ``autolife_spherized.urdf`` is
checked against the real meshes (``autolife_simple.urdf``):

  1. Sample random body configurations inside the joint limits (half of
     them in a deep squat, where the arms reach the base and shins) and
     record, per pair, the sphere-model and mesh-model distances.
  2. For every movable pair whose meshes came within 10 cm but never
     touched, locally minimise the mesh distance over the joints between
     the two links.
  3. Disable a pair only if it is
       Adjacent — rigidly attached or joined by one movable joint,
       Default  — its spheres overlap at a named pose (home, squat, or the
                  all-zero rest pose), so enabling it would make that
                  pose invalid, or
       Never    — its meshes never come within ``REACH`` of each other, or
                  never touch while at most two joints separate the links
                  (a search that small is exhaustive, so a gap left at the
                  joint limit is the designed clearance).
     Everything else stays collision-checked.

Pairs forced off by ``Default`` although their meshes can touch are
printed as sphere-model limitations.  The script rewrites the
``disable_collisions`` entries of both SRDF copies (entries for links
without spheres are kept); run ``pixi run generate-fk`` afterwards.

    pixi run audit-collisions            # 240k samples, ~5 min on 24 cores
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pinocchio as pin
from fire import Fire
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "autolife_planning" / "resources" / "robot" / "autolife"
SRDFS = [
    PKG / "autolife.srdf",
    ROOT / "resources" / "robot" / "autolife" / "autolife.srdf",
]
REACH = 0.02  # m: meshes this close are treated as able to collide
MESH_CHECK_RANGE = 0.15  # m: mesh distance is evaluated when spheres are this close
SQUAT = {"Joint_Ankle": -1.20, "Joint_Knee": -2.40, "Joint_Waist_Pitch": -1.20}


class Robot:
    """Sphere model (planner geometry) + mesh model (ground truth)."""

    def __init__(self):
        from autolife_planning.autolife import HOME_JOINTS, autolife_robot_config

        # Spheres, grouped by link.
        self.sm, sg, _ = pin.buildModelsFromUrdf(
            str(PKG / "autolife_spherized.urdf"), str(PKG)
        )
        self.sd = self.sm.createData()
        geoms = sorted(
            sg.geometryObjects, key=lambda g: self.sm.frames[g.parentFrame].name
        )
        link_of = [self.sm.frames[g.parentFrame].name for g in geoms]
        self.links = sorted(set(link_of))
        self.sph_joint = np.array([g.parentJoint for g in geoms])
        self.sph_local = np.array([g.placement.translation for g in geoms])
        self.sph_r = np.array([g.geometry.radius for g in geoms])
        self.starts = np.searchsorted(link_of, self.links)
        # Meshes, one collision pair per link pair.
        self.mm, self.mg, _ = pin.buildModelsFromUrdf(
            str(PKG / "autolife_simple.urdf"), str(PKG)
        )
        self.md = self.mm.createData()
        geom_of = {}
        for i, g in enumerate(self.mg.geometryObjects):
            geom_of.setdefault(self.mm.frames[g.parentFrame].name, i)
        self.pairs = {}
        for a in range(len(self.links)):
            for b in range(a + 1, len(self.links)):
                self.mg.addCollisionPair(
                    pin.CollisionPair(geom_of[self.links[a]], geom_of[self.links[b]])
                )
                self.pairs[a, b] = len(self.mg.collisionPairs) - 1
        self.mgd = pin.GeometryData(self.mg)
        # Body joints (everything movable except the virtual base).
        self.body = [n for n in self.mm.names[1:] if n in self.sm.names]
        ids = [self.mm.getJointId(n) for n in self.body]
        self.lo = self.mm.lowerPositionLimit[[self.mm.joints[j].idx_q for j in ids]]
        self.hi = self.mm.upperPositionLimit[[self.mm.joints[j].idx_q for j in ids]]
        self.m_q = np.array([self.mm.joints[j].idx_q for j in ids])
        self.s_q = np.array(
            [self.sm.joints[self.sm.getJointId(n)].idx_q for n in self.body]
        )
        home = dict(zip(autolife_robot_config.joint_names, HOME_JOINTS))
        self.poses = {"home": np.array([home[n] for n in self.body])}
        self.poses["squat"] = np.array([SQUAT.get(n, home[n]) for n in self.body])
        self.poses["zero"] = np.zeros(len(self.body))
        self.kind = self._classify()

    def _classify(self):
        """'adjacent' (rigid or one movable joint apart) or 'movable' per pair."""
        parent = {}
        for j in ET.parse(PKG / "autolife_spherized.urdf").getroot().findall("joint"):
            parent[j.find("child").get("link")] = (
                j.find("parent").get("link"),
                j.get("type"),
            )

        def body(link):  # climb fixed joints to the rigid body's root link
            while link in parent and parent[link][1] == "fixed":
                link = parent[link][0]
            return link

        adjacent = {
            frozenset((body(c), body(p)))
            for c, (p, t) in parent.items()
            if t != "fixed"
        }
        kind = {}
        for a, b in self.pairs:
            ba, bb = body(self.links[a]), body(self.links[b])
            kind[a, b] = (
                "adjacent" if ba == bb or frozenset((ba, bb)) in adjacent else "movable"
            )
        return kind

    def sphere_dist(self, qb):
        """Link x link minimum signed distance between spheres."""
        q = pin.neutral(self.sm)
        q[self.s_q] = qb
        pin.forwardKinematics(self.sm, self.sd, q)
        R = np.array([M.rotation for M in self.sd.oMi])[self.sph_joint]
        c = np.einsum("nij,nj->ni", R, self.sph_local)
        c += np.array([M.translation for M in self.sd.oMi])[self.sph_joint]
        D = (
            np.linalg.norm(c[:, None] - c[None], axis=-1)
            - self.sph_r[:, None]
            - self.sph_r[None]
        )
        return np.minimum.reduceat(
            np.minimum.reduceat(D, self.starts, 0), self.starts, 1
        )

    def place(self, qb):
        """Pose the mesh model; call before mesh_dist."""
        q = pin.neutral(self.mm)
        q[self.m_q] = qb
        pin.updateGeometryPlacements(self.mm, self.md, self.mg, self.mgd, q)

    def mesh_dist(self, a, b):
        k = self.pairs[a, b]
        if pin.computeCollision(self.mg, self.mgd, k):
            return 0.0
        return pin.computeDistance(self.mg, self.mgd, k).min_distance

    def chain(self, a, b):
        """Body-joint indices that move link a relative to link b."""
        ja = self.mm.frames[self.mm.getFrameId(self.links[a])].parentJoint
        jb = self.mm.frames[self.mm.getFrameId(self.links[b])].parentJoint
        names = {
            self.mm.names[j]
            for j in set(self.mm.supports[ja]) ^ set(self.mm.supports[jb])
        }
        return [i for i, n in enumerate(self.body) if n in names]


_robot: Robot | None = None


def _init():
    global _robot
    _robot = Robot()


def _sample(args):
    """Closest mesh distance (and its configuration) per movable pair."""
    seed, n = args
    r, rng = _robot, np.random.default_rng(seed)
    best = {}
    for i in range(n):
        qb = rng.uniform(r.lo, r.hi)
        if i % 2:  # deep squat: arms reach the base and the shins
            qb[[r.body.index(j) for j in SQUAT]] = rng.uniform(
                [-1.5, -2.8, -1.5], [-0.8, -1.8, 0.0]
            )
        D = r.sphere_dist(qb)
        r.place(qb)
        for (a, b), kind in r.kind.items():
            if kind == "movable" and D[a, b] < MESH_CHECK_RANGE:
                d = r.mesh_dist(a, b)
                if d < best.get((a, b), (np.inf,))[0]:
                    best[a, b] = (d, qb)
    return best


def _refine(args):
    """Locally minimise the mesh distance of one pair over its chain joints."""
    (a, b), (d0, q0) = args
    r, idx, q = _robot, _robot.chain(*args[0]), q0.copy()

    def f(x):
        q[idx] = x
        r.place(q)
        return r.mesh_dist(a, b)

    res = minimize(
        f,
        q0[idx],
        method="Powell",
        bounds=list(zip(r.lo[idx], r.hi[idx])),
        options={"maxfev": 600, "xtol": 1e-3, "ftol": 1e-4},
    )
    return (a, b), min(d0, float(res.fun))


def main(samples: int = 240_000, workers: int = 24):
    robot = Robot()
    chunks = 4 * workers
    with Pool(workers, initializer=_init) as pool:
        closest = {}
        for part in pool.map(_sample, [(s, samples // chunks) for s in range(chunks)]):
            for k, v in part.items():
                if v[0] < closest.get(k, (np.inf,))[0]:
                    closest[k] = v
        todo = [(k, v) for k, v in closest.items() if 0 < v[0] < 0.10]
        refined = dict(pool.map(_refine, todo))
    dist = {k: refined.get(k, v[0]) for k, v in closest.items()}

    at_pose = {name: robot.sphere_dist(q) for name, q in robot.poses.items()}
    disabled, limited = {}, []
    for (a, b), kind in robot.kind.items():
        la, lb = robot.links[a], robot.links[b]
        near = dist.get((a, b), np.inf)
        if kind == "adjacent":
            disabled[la, lb] = "Adjacent"
        elif any(D[a, b] < 0 for D in at_pose.values()):
            disabled[la, lb] = "Default"
            if near <= REACH:
                limited.append((la, lb, [p for p, D in at_pose.items() if D[a, b] < 0]))
        elif near > REACH or (near > 0 and len(robot.chain(a, b)) <= 2):
            disabled[la, lb] = "Never"
    n = len(robot.kind)
    print(
        f"{n} link pairs: {n - len(disabled)} checked, {len(disabled)} disabled "
        f"({', '.join(f'{w} {list(disabled.values()).count(w)}' for w in ('Adjacent', 'Default', 'Never'))})"
    )
    print(
        "Sphere-model limitations (meshes can touch, spheres overlap at a named pose):"
    )
    for la, lb, poses in limited:
        print(f"  {la} <-> {lb}  [{', '.join(poses)}]")

    links = set(robot.links)
    for path in SRDFS:
        text = path.read_text()
        keep = [
            m
            for m in re.finditer(
                r'  <disable_collisions link1="([^"]+)" link2="([^"]+)"[^\n]*\n', text
            )
            if not (m.group(1) in links and m.group(2) in links)
        ]
        entries = [m.group(0) for m in keep] + [
            f'  <disable_collisions link1="{a}" link2="{b}" reason="{w}" />\n'
            for (a, b), w in disabled.items()
        ]
        body = re.sub(r"  <disable_collisions [^\n]*\n", "", text).replace(
            "</robot>", ""
        )
        path.write_text(body + "".join(sorted(entries)) + "</robot>\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    Fire(main)
