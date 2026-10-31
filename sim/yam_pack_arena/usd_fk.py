"""Forward kinematics read straight from the written USD, with no MuJoCo and no Isaac.

This is the CPU check that ``yam.usda`` means the same arm as the MJCF: it rebuilds each body's pose
from the USD joints alone (``localPos0/Rot0``, the joint value about X, ``localPos1/Rot1``^-1) and
``tests/test_sim_yam_arena.py`` compares that with MuJoCo's ``xpos``/``xmat`` at random joint values.
It also checks that the authored body poses agree with the joints at q = 0, which PhysX relies on
when it builds the articulation (a mismatch makes the arm snap on the first step).

    python -m sim.yam_pack_arena.usd_fk        # prints the worst position and angle differences
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
YAM_USD = HERE / "assets" / "yam.usda"

# Named envelope points as (body, offset in that body's frame); the MJCF's ``grasp_site`` sits
# 0.1347 m along link_6's z. Same names and meanings as ``sim/yam_pack/spec.yaml``.
POINTS = {"tool": ("link_6", (0.0, 0.0, 0.1347)), "wrist": ("link_5", (0.0, 0.0, 0.0)),
          "elbow": ("link_3", (0.0, 0.0, 0.0))}


def _rot(q) -> np.ndarray:
    """Gf.Quat (w, xyz) to a 3x3 matrix."""
    w = q.GetReal()
    x, y, z = q.GetImaginary()
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _T(pos, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(pos, float)
    return T


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@dataclass
class Joint:
    name: str
    kind: str          # "revolute" | "prismatic"
    parent: str
    child: str
    T0: np.ndarray     # joint frame in parent
    T1: np.ndarray     # joint frame in child
    lower: float       # radians or metres
    upper: float


class UsdArm:
    def __init__(self, path: Path = YAM_USD):
        self.stage = Usd.Stage.Open(str(path))
        root = self.stage.GetDefaultPrim()
        self.root = root.GetPath().pathString
        if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError("default prim has no ArticulationRootAPI")
        self.bodies: dict[str, np.ndarray] = {}     # authored pose at q = 0, world (stage) frame
        self.mass: dict[str, float] = {}
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                M = np.array(xf).T          # Gf matrices are row-vector convention
                self.bodies[prim.GetName()] = M
                self.mass[prim.GetName()] = float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())
        self.joints: list[Joint] = []
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                rev = prim.IsA(UsdPhysics.RevoluteJoint)
                J = UsdPhysics.RevoluteJoint(prim) if rev else UsdPhysics.PrismaticJoint(prim)
                if J.GetAxisAttr().Get() != "X":
                    raise ValueError(f"{prim.GetName()}: expected axis X")
                b0 = J.GetBody0Rel().GetTargets()[0].name
                b1 = J.GetBody1Rel().GetTargets()[0].name
                lo, hi = J.GetLowerLimitAttr().Get(), J.GetUpperLimitAttr().Get()
                if rev:
                    lo, hi = np.radians(lo), np.radians(hi)
                self.joints.append(Joint(
                    prim.GetName(), "revolute" if rev else "prismatic", b0, b1,
                    _T(J.GetLocalPos0Attr().Get(), _rot(J.GetLocalRot0Attr().Get())),
                    _T(J.GetLocalPos1Attr().Get(), _rot(J.GetLocalRot1Attr().Get())),
                    float(lo), float(hi)))
        children = {j.child for j in self.joints}
        roots = [b for b in self.bodies if b not in children]
        if len(roots) != 1:
            raise ValueError(f"expected one root body, found {roots}")
        self.base = roots[0]
        self.by_child = {j.child: j for j in self.joints}

    def fk(self, q: dict[str, float], base: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Each body's 4x4 world pose for joint values ``q`` (missing joints at 0)."""
        out = {self.base: self.bodies[self.base] if base is None else base}

        def pose(b: str) -> np.ndarray:
            if b in out:
                return out[b]
            j = self.by_child[b]
            v = float(q.get(j.name, 0.0))
            M = np.eye(4)
            if j.kind == "revolute":
                M[:3, :3] = _rx(v)
            else:
                M[0, 3] = v
            out[b] = pose(j.parent) @ j.T0 @ M @ np.linalg.inv(j.T1)
            return out[b]

        for b in self.bodies:
            pose(b)
        return out

    def points(self, q: dict[str, float], base: np.ndarray | None = None) -> dict[str, np.ndarray]:
        poses = self.fk(q, base)
        return {n: (poses[b] @ np.array([*off, 1.0]))[:3] for n, (b, off) in POINTS.items()}

    def rest_mismatch(self) -> tuple[float, float]:
        """Largest position (m) and angle (rad) gap between the authored body poses and FK at q = 0."""
        fk0 = self.fk({})
        dp = max(float(np.linalg.norm(fk0[b][:3, 3] - M[:3, 3])) for b, M in self.bodies.items())
        da = max(_angle(fk0[b][:3, :3], M[:3, :3]) for b, M in self.bodies.items())
        return dp, da


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Rotation angle between two frames, accurate for small angles (arccos of the trace is not)."""
    f = float(np.linalg.norm(Ra - Rb)) / (2 * np.sqrt(2))
    return float(2 * np.arcsin(min(1.0, f)))


def compare_with_mujoco(mjcf: Path, usd: Path = YAM_USD, n: int = 200, seed: int = 0) -> dict:
    """Worst body-position, body-angle and named-point gaps between USD FK and MuJoCo over ``n``
    random joint vectors within the limits; plus the joint limits and masses side by side."""
    import mujoco

    arm = UsdArm(usd)
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    worst = {"body_pos_m": 0.0, "body_angle_rad": 0.0, "point_m": 0.0}
    gsite = m.site("grasp_site").id
    for _ in range(n):
        q = {}
        d.qpos[:] = 0
        for j in arm.joints:
            jid = m.joint(j.name).id
            v = float(rng.uniform(*m.jnt_range[jid]))
            q[j.name] = v
            d.qpos[m.jnt_qposadr[jid]] = v
        mujoco.mj_kinematics(m, d)
        poses = arm.fk(q)
        for b, M in poses.items():
            bid = m.body(b).id
            worst["body_pos_m"] = max(worst["body_pos_m"], float(np.linalg.norm(M[:3, 3] - d.xpos[bid])))
            worst["body_angle_rad"] = max(worst["body_angle_rad"], _angle(M[:3, :3], d.xmat[bid].reshape(3, 3)))
        pts = arm.points(q)
        mj_pts = {"tool": d.site_xpos[gsite], "wrist": d.xpos[m.body("link_5").id],
                  "elbow": d.xpos[m.body("link_3").id]}
        for k in pts:
            worst["point_m"] = max(worst["point_m"], float(np.linalg.norm(pts[k] - mj_pts[k])))
    limits = max(max(abs(j.lower - m.jnt_range[m.joint(j.name).id][0]),
                     abs(j.upper - m.jnt_range[m.joint(j.name).id][1])) for j in arm.joints)
    # Total mass of the bodies merged into each USD body equals MuJoCo's subtree minus physical children.
    total_usd = sum(arm.mass.values())
    total_mj = float(m.body_subtreemass[m.body(arm.base).id])
    dp, da = arm.rest_mismatch()
    return {**worst, "limit_gap": float(limits), "mass_usd_kg": total_usd, "mass_mujoco_kg": total_mj,
            "rest_pos_m": dp, "rest_angle_rad": da, "samples": n}


if __name__ == "__main__":
    from sim.yam_pack_arena.usd_build import COLLISION_MJCF

    for k, v in compare_with_mujoco(COLLISION_MJCF).items():
        print(f"{k:16s} {v:.3g}" if isinstance(v, float) else f"{k:16s} {v}")
