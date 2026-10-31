"""Write the YAM arm and the pack scene's box as USD for Isaac Lab-Arena, from the same MJCF the MuJoCo scene uses.

    python -m sim.yam_pack_arena.usd_build                         # assets/yam.usda, assets/tray.usda
    python -m sim.yam_pack_arena.usd_build --visual /path/to/mujoco_menagerie/i2rt_yam/yam.xml
                                                                     # also assets/build/yam_visual.usdc (meshes)

There is no public YAM articulation for Isaac Lab-Arena (IsaacLab-Arena e87874d ships agibot, droid,
franka, g1, galbot, gr1t2 and kuka_allegro), so this builds one. The source is the MJCF the MuJoCo
scene already runs, Robocurve's collision model vendored in ``sim/yam_pack/assets`` (from
inspect-robots-yam 9b5ebb5, a reduction of Menagerie i2rt_yam at 71f066a), compiled by MuJoCo, so the
kinematics, joint ranges, inertials and collision capsules are MuJoCo's own numbers rather than a
second reading of the XML:

- one rigid body per MuJoCo body that has a joint, plus the base; jointless bodies (the finger
  sub-frames ``lf_rot``, ``lf_down``, ...) are merged into the body they hang from, masses and
  inertias combined about the merged centre of mass;
- each body prim sits at its MuJoCo body frame at q = 0, so Isaac's ``body_link_pose_w`` is MuJoCo's
  ``xpos``/``xquat`` and the named envelope points mean the same thing in both simulators;
- each hinge or slide becomes a ``PhysicsRevoluteJoint`` / ``PhysicsPrismaticJoint`` about its local X,
  with the joint frame built so that a positive joint value turns the same way as in MuJoCo, limits
  from ``jnt_range`` (degrees for revolute, as USD Physics requires);
- capsules, spheres and boxes become ``UsdGeom.Capsule``/``Sphere``/``Cube`` colliders.

What is not here and is set at spawn time by the Isaac Lab config (``embodiment.py``): the drive gains,
armature and joint friction (MuJoCo's ``frictionloss``), gravity off for the arm (MuJoCo's ``gravcomp``),
and the fixed base. ``usd_fk.py`` reads the written USD back and checks its forward kinematics against
MuJoCo's at random joint values; ``check_usd.py`` does the same inside Isaac on Brev.

The visual model (optional) uses Menagerie i2rt_yam's meshes; it is not committed (5 MB of meshes) and
only matters for rendered clips. Its kinematics are checked against the collision model's before it is
written.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from sim.yam_pack.spec import load_spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COLLISION_MJCF = REPO / "sim/yam_pack/assets/yam_collision.xml"
ASSETS = HERE / "assets"
ROOT = "/yam"
MU = 1.0   # MuJoCo's default sliding friction


# ----- small rotation helpers (MuJoCo quaternions are w, x, y, z) -------------------------------
def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q / np.linalg.norm(q)


def quat_to_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def gf_quat(q) -> Gf.Quatf:
    w, x, y, z = (float(v) for v in q)
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def frame_with_x(axis: np.ndarray) -> np.ndarray:
    """A right-handed rotation whose first column is ``axis``. Deterministic: the second column is
    the world axis least aligned with ``axis``, orthogonalised."""
    x = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.eye(3)[int(np.argmin(np.abs(x)))]
    y = helper - x * float(helper @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


# ----- the compiled model at q = 0 ------------------------------------------------------------------
class Compiled:
    """MuJoCo's compiled model at q = 0 with the body groups the USD will have."""

    def __init__(self, mjcf: Path):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        m = self.m
        # A body is physical if it is the root body (child of world) or carries a joint.
        self.physical = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 or m.body_jntnum[b] > 0]
        self.owner = {}
        for b in range(1, m.nbody):
            o = b
            while o not in self.physical:
                o = m.body_parentid[o]
            self.owner[b] = o

    def name(self, b: int) -> str:
        return self.m.body(b).name

    def pose(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def group(self, p: int) -> list[int]:
        return [b for b in range(1, self.m.nbody) if self.owner[b] == p]

    def mass_props(self, p: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Mass, centre of mass in p's frame, principal moments and principal-axes quaternion in p's
        frame, for p and every jointless body merged into it."""
        m, d = self.m, self.d
        bodies = [b for b in self.group(p) if m.body_mass[b] > 0]
        total = float(sum(m.body_mass[b] for b in bodies))
        pp, Rp = self.pose(p)
        if total <= 0:
            return 0.0, np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0])
        com = sum(m.body_mass[b] * d.xipos[b] for b in bodies) / total
        inertia = np.zeros((3, 3))
        for b in bodies:
            Ri = d.ximat[b].reshape(3, 3)
            Ib = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
            r = d.xipos[b] - com
            inertia += Ib + m.body_mass[b] * (float(r @ r) * np.eye(3) - np.outer(r, r))
        local = Rp.T @ inertia @ Rp
        moments, axes = np.linalg.eigh(local)
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        return total, Rp.T @ (com - pp), moments, mat_to_quat(axes)


# ----- USD writers -------------------------------------------------------------------------------
def _new_stage(path: Path) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def _set_pose(xf: UsdGeom.Xformable, pos, quat_wxyz, scale=None) -> None:
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in pos)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(gf_quat(quat_wxyz))
    if scale is not None:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(*(float(v) for v in scale)))


def _add_shape(stage, path: str, gtype: int, size: np.ndarray, pos, quat, collider: bool, rgba=None):
    """A MuJoCo primitive as a USD gprim at ``pos``/``quat`` in its body's frame."""
    T = mujoco.mjtGeom
    if gtype == T.mjGEOM_CAPSULE:
        g = UsdGeom.Capsule.Define(stage, path)
        g.CreateRadiusAttr(float(size[0]))
        g.CreateHeightAttr(float(2 * size[1]))   # USD height is the cylinder part, like MuJoCo's 2 x half-length
        g.CreateAxisAttr(UsdGeom.Tokens.z)
        _set_pose(g, pos, quat)
    elif gtype == T.mjGEOM_SPHERE:
        g = UsdGeom.Sphere.Define(stage, path)
        g.CreateRadiusAttr(float(size[0]))
        _set_pose(g, pos, quat)
    elif gtype == T.mjGEOM_BOX:
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(2.0)                    # extent +-1, scaled to MuJoCo's half-sizes
        _set_pose(g, pos, quat, scale=size[:3])
    elif gtype == T.mjGEOM_CYLINDER:
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(float(size[0]))
        g.CreateHeightAttr(float(2 * size[1]))
        g.CreateAxisAttr(UsdGeom.Tokens.z)
        _set_pose(g, pos, quat)
    else:
        raise ValueError(f"geom type {gtype} at {path} has no USD primitive here")
    if rgba is not None:
        g.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in rgba[:3]))])
    if collider:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


def write_robot(c: Compiled, out: Path, visual: Compiled | None = None) -> Path:
    """The YAM arm as one articulation rooted at ``/yam``. Collision geometry from ``c``; visual meshes
    from ``visual`` when given (its bodies must sit where ``c``'s do, checked by the caller)."""
    m, d = c.m, c.d
    stage = _new_stage(out)
    root = UsdGeom.Xform.Define(stage, ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("source", f"{COLLISION_MJCF.relative_to(REPO)} compiled by MuJoCo {mujoco.__version__}")

    for p in c.physical:
        name = c.name(p)
        pp, Rp = c.pose(p)
        body = UsdGeom.Xform.Define(stage, f"{ROOT}/{name}")
        _set_pose(body, pp, mat_to_quat(Rp))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        _no_gravity(body.GetPrim())
        mass, com, moments, axes = c.mass_props(p)
        mapi = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mapi.CreateMassAttr(mass)
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in com)))
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in moments)))
        mapi.CreatePrincipalAxesAttr(gf_quat(axes))
        # Colliders: every geom of the merged group, placed in p's frame.
        UsdGeom.Xform.Define(stage, f"{ROOT}/{name}/collisions")
        k = 0
        for b in c.group(p):
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    continue   # a collision model has none; visual meshes come from ``visual``
                gp, gR = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
                _add_shape(stage, f"{ROOT}/{name}/collisions/c{k}", m.geom_type[g], m.geom_size[g],
                           Rp.T @ (gp - pp), mat_to_quat(Rp.T @ gR), collider=True)
                k += 1
        if visual is not None:
            _add_visual_meshes(stage, visual, name, pp, Rp)
            cprim = stage.GetPrimAtPath(f"{ROOT}/{name}/collisions")
            if cprim.IsValid():
                UsdGeom.Imageable(cprim).CreatePurposeAttr(UsdGeom.Tokens.guide)

    UsdGeom.Scope.Define(stage, f"{ROOT}/joints")
    for j in range(m.njnt):
        b = m.jnt_bodyid[j]
        parent = c.owner[m.body_parentid[b]]
        if parent == 0 or c.owner.get(parent) is None:
            raise ValueError(f"joint {m.joint(j).name} has no parent body to attach to")
        anchor, axis = d.xanchor[j], d.xaxis[j]
        Rj = frame_with_x(axis)
        pp, Rp = c.pose(parent)
        pb, Rb = c.pose(b)
        jt = m.jnt_type[j]
        path = f"{ROOT}/joints/{m.joint(j).name}"
        if jt == mujoco.mjtJoint.mjJNT_HINGE:
            J = UsdPhysics.RevoluteJoint.Define(stage, path)
            lo, hi = (math.degrees(v) for v in m.jnt_range[j])
            drive_kind = "angular"
        elif jt == mujoco.mjtJoint.mjJNT_SLIDE:
            J = UsdPhysics.PrismaticJoint.Define(stage, path)
            lo, hi = (float(v) for v in m.jnt_range[j])
            drive_kind = "linear"
        else:
            raise ValueError(f"joint type {jt} not supported")
        if not m.jnt_limited[j]:
            raise ValueError(f"joint {m.joint(j).name} is unlimited; the YAM's are all limited")
        J.CreateAxisAttr("X")
        J.CreateLowerLimitAttr(float(lo))
        J.CreateUpperLimitAttr(float(hi))
        J.CreateBody0Rel().SetTargets([Sdf.Path(f"{ROOT}/{c.name(parent)}")])
        J.CreateBody1Rel().SetTargets([Sdf.Path(f"{ROOT}/{c.name(b)}")])
        J.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in Rp.T @ (anchor - pp))))
        J.CreateLocalRot0Attr(gf_quat(mat_to_quat(Rp.T @ Rj)))
        J.CreateLocalPos1Attr(Gf.Vec3f(*(float(v) for v in Rb.T @ (anchor - pb))))
        J.CreateLocalRot1Attr(gf_quat(mat_to_quat(Rb.T @ Rj)))
        drive = UsdPhysics.DriveAPI.Apply(J.GetPrim(), drive_kind)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        # Gains are set by the Isaac Lab actuator config (embodiment.py); zero here so the asset alone
        # never moves on its own.
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    stage.GetRootLayer().Save()
    return out


def _no_gravity(prim) -> None:
    """MuJoCo's ``gravcomp`` on the arm, as PhysX's per-body gravity switch. Written as the raw
    PhysxSchema attribute because usd-core has no PhysxSchema; Isaac reads it like any other."""
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)


def _friction_material(stage, path: str, mu: float) -> UsdShade.Material:
    """MuJoCo's default sliding friction (1.0) for every contact; PhysX's default is 0.5. PhysX
    combines two materials by averaging by default, so both sides carry the same value."""
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_physics_material(prim, mat: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _add_visual_meshes(stage, v: Compiled, body_name: str, pp: np.ndarray, Rp: np.ndarray) -> None:
    m, d = v.m, v.d
    bid = m.body(body_name).id
    UsdGeom.Xform.Define(stage, f"{ROOT}/{body_name}/visuals")
    k = 0
    for b in v.group(bid):
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            faces = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            gp, gR = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            mesh = UsdGeom.Mesh.Define(stage, f"{ROOT}/{body_name}/visuals/v{k}")
            mesh.CreatePointsAttr([Gf.Vec3f(*(float(x) for x in p)) for p in verts])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in m.geom_rgba[g][:3]))])
            _set_pose(mesh, Rp.T @ (gp - pp), mat_to_quat(Rp.T @ gR))
            k += 1


def write_tray(spec: dict, out: Path) -> Path:
    """The open-top box as one rigid body: floor and four walls, origin at the floor's centre on the
    table, exactly the MuJoCo scene's ``box`` body (so its pose means the same thing to the oracle)."""
    box = spec["objects"]["box"]
    (sx, sy, sz), t = box["size"], box["wall"]
    walls = [
        ((0, 0, t / 2), (sx / 2, sy / 2, t / 2)),
        ((sx / 2 - t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ((-sx / 2 + t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ((0, sy / 2 - t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
        ((0, -sy / 2 + t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
    ]
    vol = sum(8 * h[0] * h[1] * h[2] for _, h in walls)
    stage = _new_stage(out)
    root = UsdGeom.Xform.Define(stage, "/tray")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mapi = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mapi.CreateMassAttr(float(box["mass"]))
    # Same uniform density as the MuJoCo box, so the centre of mass and inertia come out the same.
    com = sum(np.array(p) * 8 * h[0] * h[1] * h[2] for p, h in walls) / vol
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in com)))
    inertia = np.zeros(3)
    rho = box["mass"] / vol
    for p, h in walls:
        mw = rho * 8 * h[0] * h[1] * h[2]
        a, b, c_ = 2 * h[0], 2 * h[1], 2 * h[2]
        r = np.array(p) - com
        inertia += mw / 12 * np.array([b * b + c_ * c_, a * a + c_ * c_, a * a + b * b])
        inertia += mw * np.array([r[1] ** 2 + r[2] ** 2, r[0] ** 2 + r[2] ** 2, r[0] ** 2 + r[1] ** 2])
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in inertia)))
    mat = _friction_material(stage, "/tray/physics_material", MU)
    for i, (p, h) in enumerate(walls):
        g = _add_shape(stage, f"/tray/wall{i}", mujoco.mjtGeom.mjGEOM_BOX, np.array(h), p, (1, 0, 0, 0),
                       collider=True, rgba=(0.55, 0.4, 0.25, 1))
        _bind_physics_material(g.GetPrim(), mat)
    stage.GetRootLayer().Save()
    return out


def write_block(spec: dict, out: Path) -> Path:
    """One block: a cube with its origin at its centre, like the MuJoCo scene's ``block{i}`` bodies.
    The same asset is spawned three times."""
    b = spec["objects"]["blocks"]
    s = float(b["size"])
    stage = _new_stage(out)
    root = UsdGeom.Xform.Define(stage, "/block")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mapi = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mapi.CreateMassAttr(float(b["mass"]))
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    i = float(b["mass"]) * s * s / 6
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(i, i, i))
    g = _add_shape(stage, "/block/cube", mujoco.mjtGeom.mjGEOM_BOX, np.array([s / 2] * 3), (0, 0, 0), (1, 0, 0, 0),
                   collider=True, rgba=(0.85, 0.2, 0.2, 1))
    _bind_physics_material(g.GetPrim(), _friction_material(stage, "/block/physics_material", MU))
    stage.GetRootLayer().Save()
    return out


def write_hand(spec: dict, out: Path) -> Path:
    """The scripted hand: a kinematic sphere with no collider (MuJoCo's mocap body with
    contype 0), moved by the driver. It is a marker; the proximity sensor and the trace use the
    driver's own hand position, never this prim's."""
    stage = _new_stage(out)
    root = UsdGeom.Xform.Define(stage, "/hand")
    stage.SetDefaultPrim(root.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1)
    _add_shape(stage, "/hand/sphere", mujoco.mjtGeom.mjGEOM_SPHERE, np.array([spec["hand"]["radius_display"], 0, 0]),
               (0, 0, 0), (1, 0, 0, 0), collider=False, rgba=(0.95, 0.75, 0.6, 1))
    stage.GetRootLayer().Save()
    return out


def write_table(out: Path, half: float = 1.0, thickness: float = 0.05) -> Path:
    """The table: a static box whose top face is z = 0, standing in for MuJoCo's ground plane
    (Isaac's own ground plane loads from Nucleus, which a bare Brev box may not reach)."""
    stage = _new_stage(out)
    root = UsdGeom.Xform.Define(stage, "/table")
    stage.SetDefaultPrim(root.GetPrim())
    g = _add_shape(stage, "/table/top", mujoco.mjtGeom.mjGEOM_BOX, np.array([half, half, thickness / 2]),
                   (0, 0, -thickness / 2), (1, 0, 0, 0), collider=True, rgba=(0.8, 0.75, 0.65, 1))
    _bind_physics_material(g.GetPrim(), _friction_material(stage, "/table/physics_material", MU))
    stage.GetRootLayer().Save()
    return out


def check_same_kinematics(a: Compiled, b: Compiled, names: list[str], n: int = 20, tol: float = 1e-6) -> float:
    """Largest body-position difference between two models over random joint values (by joint name)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    joints = [a.m.joint(j).name for j in range(a.m.njnt)]
    for _ in range(n):
        for c in (a, b):
            c.d.qpos[:] = 0
        for jn in joints:
            ja = a.m.joint(jn).id
            lo, hi = a.m.jnt_range[ja]
            v = rng.uniform(lo, hi)
            a.d.qpos[a.m.jnt_qposadr[ja]] = v
            b.d.qpos[b.m.jnt_qposadr[b.m.joint(jn).id]] = v
        for c in (a, b):
            mujoco.mj_kinematics(c.m, c.d)
        for nm in names:
            worst = max(worst, float(np.linalg.norm(a.d.xpos[a.m.body(nm).id] - b.d.xpos[b.m.body(nm).id])))
    for c in (a, b):
        c.d.qpos[:] = 0
        mujoco.mj_forward(c.m, c.d)
    if worst > tol:
        raise ValueError(f"the visual model's bodies sit up to {worst:.2e} m from the collision model's")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visual", type=Path, help="Menagerie i2rt_yam/yam.xml, for a meshed copy (assets/build/)")
    ap.add_argument("--out", type=Path, default=ASSETS)
    args = ap.parse_args(argv)
    spec = load_spec()
    col = Compiled(COLLISION_MJCF)
    print("wrote", write_robot(col, args.out / "yam.usda"))
    print("wrote", write_tray(spec, args.out / "tray.usda"))
    print("wrote", write_block(spec, args.out / "block.usda"))
    print("wrote", write_hand(spec, args.out / "hand.usda"))
    print("wrote", write_table(args.out / "table.usda"))
    if args.visual:
        vis = Compiled(args.visual)
        worst = check_same_kinematics(col, vis, [col.name(p) for p in col.physical])
        print(f"visual model kinematics match the collision model's (worst {worst:.1e} m)")
        print("wrote", write_robot(col, args.out / "build/yam_visual.usdc", visual=vis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
