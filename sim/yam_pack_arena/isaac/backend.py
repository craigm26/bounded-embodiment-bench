"""The Isaac Lab side of ``core.PackDriver``: the ``Backend`` over Isaac Lab assets, the action term
that runs the driver inside Isaac Lab's env step, the observation term, and the start-up event
that keeps the arms from colliding with the box and blocks (as in MuJoCo, where they never touch).

Needs Isaac Lab (pinned by Arena e87874d to Isaac Lab e3f7d9ff); not importable on a CPU-only box.
One environment per process: the driver, the hand script and the trace are per episode, and verdict
runs want PhysX on the CPU device for repeatability (``BREV.md``).
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from sim.yam_pack.episode import SIDES
from sim.yam_pack.spec import load_spec
from sim.yam_pack_arena.core import PackDriver
from sim.yam_pack.episode import draw

ARM_JOINTS = [f"joint{j}" for j in range(1, 7)]
FINGERS = ("left_finger", "right_finger")
TOOL_OFFSET = (0.0, 0.0, 0.1347)       # grasp_site in link_6, from the MJCF (usd_fk.POINTS)
OBJECTS = ("box", "block0", "block1", "block2")
SCENE_KEYS = {"box": "tray", "block0": "block0", "block1": "block1", "block2": "block2"}


def _t(x) -> torch.Tensor:
    """Isaac Lab 3 data properties are ProxyArrays over warp arrays; older ones are torch tensors."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "torch"):
        return x.torch
    import warp as wp

    return wp.to_torch(x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().double().numpy()


def xyzw_to_wxyz(q) -> np.ndarray:
    q = np.asarray(q, float)
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q) -> np.ndarray:
    q = np.asarray(q, float)
    return np.array([q[1], q[2], q[3], q[0]])


def _rotate(q_wxyz, v) -> np.ndarray:
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ np.asarray(v, float)


def _call(obj, names: tuple[str, ...], **kwargs):
    """Call the first method of ``obj`` that exists, for API names that moved between Isaac Lab
    releases (Arena e87874d pins one; this keeps a newer pin from failing on a rename alone)."""
    for n in names:
        fn = getattr(obj, n, None)
        if fn is not None:
            return fn(**kwargs)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


class IsaacBackend:
    """``core.Backend`` over one Isaac Lab environment (index 0). Positions are returned in the
    environment's own frame (world minus ``env_origins[0]``), which is the table frame."""

    def __init__(self, env, spec: dict):
        self.env, self.spec = env, spec
        sc = env.scene
        self.arm = {s: sc[f"{s}_robot"] for s in SIDES}
        self.obj = {o: sc[SCENE_KEYS[o]] for o in OBJECTS}
        self.hand_obj = sc["hand"]
        self.physics_dt = float(env.physics_dt)
        self.device = env.device
        self.origin = _np(sc.env_origins)[0]
        self.env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self.jids = {s: self.arm[s].find_joints(ARM_JOINTS, preserve_order=True)[0] for s in SIDES}
        self.fids = {s: self.arm[s].find_joints(list(FINGERS), preserve_order=True)[0] for s in SIDES}
        pts = spec["robot"]["points"]
        self.bid = {}
        for s in SIDES:
            names = ["link_6", pts["wrist"]["body"], pts["elbow"]["body"]]
            ids = self.arm[s].find_bodies(names, preserve_order=True)[0]
            self.bid[s] = dict(zip(names, ids))
        box = spec["objects"]["box"]
        (_, sy, sz), t = box["size"], box["wall"]
        self.grip_offset = {"box": (0.0, sy / 2 - t / 2, sz)}
        for o in OBJECTS[1:]:
            self.grip_offset[o] = (0.0, 0.0, spec["objects"]["blocks"]["size"] / 2)
        self.pt_ref = {f"{s}_{p}": (s, ref) for s in SIDES for p, ref in pts.items()}

    # ----- reads ---------------------------------------------------------------------------------
    def _body_pose(self, side: str, body: str) -> tuple[np.ndarray, np.ndarray]:
        pose = _np(self.arm[side].data.body_link_pose_w)[0, self.bid[side][body]]
        return pose[:3] - self.origin, xyzw_to_wxyz(pose[3:7])

    def joint_q(self, side):
        return _np(self.arm[side].data.joint_pos)[0, self.jids[side]]

    def point(self, pid):
        side, ref = self.pt_ref[pid]
        if "site" in ref:      # the tool: grasp_site on link_6
            p, q = self._body_pose(side, "link_6")
            return p + _rotate(q, TOOL_OFFSET)
        return self._body_pose(side, ref["body"])[0]

    def object_pose(self, name):
        pose = _np(self.obj[name].data.root_link_pose_w)[0]
        return pose[:3] - self.origin, xyzw_to_wxyz(pose[3:7])

    def grip_point(self, name):
        p, q = self.object_pose(name)
        return p + _rotate(q, self.grip_offset[name])

    def link6_state(self, side):
        p, q = self._body_pose(side, "link_6")
        vel = _np(self.arm[side].data.body_link_vel_w)[0, self.bid[side]["link_6"]]
        return p, q, vel[:3], vel[3:6]

    def arm_contacts(self):
        # MuJoCo lists left/right arm body pairs in contact; the envelope has no rule on it and the
        # oracle only counts it in metrics. Not wired to a contact sensor in Isaac yet.
        return []

    # ----- writes --------------------------------------------------------------------------------
    def _tensor(self, rows) -> torch.Tensor:
        return torch.tensor(np.asarray(rows, float), dtype=torch.float32, device=self.device).reshape(1, -1)

    def set_arm_target(self, side, q):
        a = self.arm[side]
        target = self._tensor(q)
        tc = getattr(getattr(a, "actuators", None), "target_command", None)
        if tc is not None and hasattr(tc, "set_position_index"):
            tc.set_position_index(value=target, joint_ids=self.jids[side], env_ids=self.env_ids)
        else:
            a.set_joint_position_target_index(target=target, joint_ids=self.jids[side], env_ids=self.env_ids)

    def set_object_state(self, name, pos, quat, linvel, angvel):
        o = self.obj[name]
        pose = self._tensor([*(np.asarray(pos) + self.origin), *wxyz_to_xyzw(quat)])
        vel = self._tensor([*linvel, *angvel])
        _call(o, ("write_root_link_pose_to_sim_index", "write_root_pose_to_sim_index"), root_pose=pose, env_ids=self.env_ids)
        _call(o, ("write_root_link_velocity_to_sim_index",), root_velocity=vel, env_ids=self.env_ids)

    def object_state(self, name):
        d = self.obj[name].data
        p, q = self.object_pose(name)
        com = _np(d.root_com_pos_w)[0] - self.origin
        return p, q, com, _np(d.root_com_lin_vel_w)[0], _np(d.root_com_ang_vel_w)[0]

    def _mass_inertia(self, name) -> tuple[float, np.ndarray]:
        """Mass and the inertia tensor in the body frame, read once from PhysX (CPU tensors)."""
        cache = self.__dict__.setdefault("_mi", {})
        if name not in cache:
            view = self.obj[name].root_physx_view
            cache[name] = (float(_np(view.get_masses())[0].sum()), _np(view.get_inertias())[0].reshape(-1)[:9].reshape(3, 3))
        return cache[name]

    def drive_object(self, name, lin_acc, ang_acc):
        """The soft weld's force and torque, through Isaac Lab's external wrench (held and applied on
        every ``write_data_to_sim`` until changed). Isaac Lab takes them in the body frame at the centre
        of mass, so the world-frame values are rotated in. Not yet run on a GPU (BREV.md)."""
        o = self.obj[name]
        if lin_acc is None:
            f = t = np.zeros(3)
        else:
            mass, I_body = self._mass_inertia(name)
            _, q, _, _, w = self.object_state(name)
            R = np.array([_rotate(q, e) for e in np.eye(3)]).T            # body to world
            g = np.asarray(self.env.sim.cfg.gravity, float)
            I = R @ I_body @ R.T
            F = mass * (np.asarray(lin_acc) - g)
            T = I @ np.asarray(ang_acc) + np.cross(w, I @ w)
            f, t = R.T @ F, R.T @ T
        z = lambda v: torch.tensor(np.asarray(v, float), dtype=torch.float32, device=self.device).reshape(1, 1, 3)  # noqa: E731
        _call(o, ("set_external_force_and_torque",), forces=z(f), torques=z(t), body_ids=[0], env_ids=self.env_ids)

    def set_hand(self, pos):
        pose = self._tensor([*(np.asarray(pos) + self.origin), 0.0, 0.0, 0.0, 1.0])
        _call(self.hand_obj, ("write_root_pose_to_sim_index",), root_pose=pose, env_ids=self.env_ids)

    # ----- episode start -------------------------------------------------------------------------
    def reset_scene(self, block_xy, hand_start) -> None:
        """Put the arms at ``start_q`` with fingers open and still, the box and blocks where the seed
        put them, and the hand at its start. Same initial state as ``scene.World``."""
        spec = self.spec
        for s in SIDES:
            a = self.arm[s]
            q0 = self._tensor(spec["robot"]["start_q"][s])
            zeros = torch.zeros_like(q0)
            a.write_joint_position_to_sim_index(position=q0, joint_ids=self.jids[s], env_ids=self.env_ids)
            a.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=self.jids[s], env_ids=self.env_ids)
            self.set_arm_target(s, spec["robot"]["start_q"][s])
            lims = _np(a.data.joint_pos_limits)[0, self.fids[s]]      # (2, 2): lower, upper per finger
            fq = self._tensor([lims[0][1], lims[1][0]])               # left finger at its upper, right at its lower
            a.write_joint_position_to_sim_index(position=fq, joint_ids=self.fids[s], env_ids=self.env_ids)
            a.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(fq), joint_ids=self.fids[s], env_ids=self.env_ids)
            tc = getattr(getattr(a, "actuators", None), "target_command", None)
            if tc is not None and hasattr(tc, "set_position_index"):
                tc.set_position_index(value=fq, joint_ids=self.fids[s], env_ids=self.env_ids)
            else:
                a.set_joint_position_target_index(target=fq, joint_ids=self.fids[s], env_ids=self.env_ids)
        box = spec["objects"]["box"]
        cx, cy, _ = box["centre"]
        zero = np.zeros(3)
        for o in OBJECTS:                      # no soft-weld wrench left over from the last episode
            self.drive_object(o, None, None)
        self.set_object_state("box", (cx, cy, 0.0), (1, 0, 0, 0), zero, zero)
        s = spec["objects"]["blocks"]["size"]
        for i, (x, y) in enumerate(block_xy):
            self.set_object_state(f"block{i}", (x, y, s / 2), (1, 0, 0, 0), zero, zero)
        self.set_hand(hand_start)


# ----- the action term --------------------------------------------------------------------------
class PackDriverAction(ActionTerm):
    """Runs ``core.PackDriver`` inside Isaac Lab's step. The 14-D action is the MuJoCo task's:
    [left joint1..6, left grip, right joint1..6, right grip], joints in radians, grip 1 open, 0 closed.

    Per env step Isaac Lab calls ``process_actions`` once, then for each of ``decimation`` substeps
    ``apply_actions``, ``write_data_to_sim``, a physics step and ``scene.update``. The driver's
    post-physics work for substep k runs at the start of substep k + 1 (``apply_actions``) and, for
    the last substep, in ``sync()``, which the observation term and the runner call.

    Set ``seed`` / ``overrides`` on the term before ``env.reset()`` to choose the episode.
    """

    cfg: "PackDriverActionCfg"

    def __init__(self, cfg: "PackDriverActionCfg", env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("the pack driver runs one environment per process (num_envs=1)")
        self.spec = load_spec()
        self.backend = IsaacBackend(env, self.spec)
        if abs(self.backend.physics_dt - self.spec["robot"]["physics_dt"]) > 1e-12:
            raise ValueError(f"physics dt {self.backend.physics_dt} != spec {self.spec['robot']['physics_dt']}")
        self.seed = cfg.seed
        self.overrides: dict | None = None
        self.driver: PackDriver | None = None
        self._draws = None
        self._stepped = False
        self._raw = torch.zeros(env.num_envs, 14, device=self.device)

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def ensure_driver(self) -> PackDriver:
        """The episode's driver, built on first use after a reset (after Isaac Lab's ``sim.forward``,
        so the first trace frame reads the reset state)."""
        if self.driver is None:
            if self._draws is None:
                self.reset()
            self.driver = PackDriver(self.backend, self.spec, self.seed, draws=self._draws)
        return self.driver

    def sync(self) -> None:
        if self._stepped and self.driver is not None:
            self.driver.post_physics()
        self._stepped = False

    def process_actions(self, actions: torch.Tensor):
        self.sync()
        self._raw[:] = actions
        self.ensure_driver().begin(actions[0].detach().cpu().double().numpy())

    def apply_actions(self):
        self.sync()
        self.driver.pre_physics()
        self._stepped = True

    def reset(self, env_ids=None) -> None:
        self.driver, self._stepped = None, False
        self._draws = draw(self.spec, self.seed, self.overrides)
        self.backend.reset_scene(self._draws.block_xy, self._draws.hand_start)


@configclass
class PackDriverActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PackDriverAction
    asset_name: str = "left_robot"
    seed: int = 0


@configclass
class PackActionsCfg:
    driver: PackDriverActionCfg = PackDriverActionCfg()


# ----- observation term ---------------------------------------------------------------------------
def pack_state(env) -> torch.Tensor:
    """Both arms' joints and the two grip states, 14 values in the action's layout. The proximity
    reading is not here: the reference policies read it once per step from ``driver.observation()``,
    as in MuJoCo, and an observation manager may compute its terms more than once per step."""
    term: PackDriverAction = env.action_manager.get_term("driver")
    term.sync()
    drv = term.ensure_driver()
    row = [*drv.q("left"), drv.grip["left"], *drv.q("right"), drv.grip["right"]]
    return torch.tensor([row], dtype=torch.float32, device=env.device)


# ----- start-up event: arms do not collide with the box or blocks ---------------------------------
def filter_arm_object_collisions(env, env_ids=None, arm_paths: list[str] = MISSING, object_paths: list[str] = MISSING):
    """``prestartup`` event: put the arm and object prims of every environment in two collision
    groups that do not collide with each other, matching the MuJoCo scene's contype/conaffinity
    (arms touch the table and each other; box and blocks touch the table and each other)."""
    from pxr import Sdf, UsdPhysics

    stage = env.scene.stage if hasattr(env.scene, "stage") else env.sim.stage
    arms = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path("/World/collision_groups/arms"))
    objs = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path("/World/collision_groups/objects"))
    for group, paths in ((arms, arm_paths), (objs, object_paths)):
        inc = group.CreateCollidersCollectionAPI().CreateIncludesRel()
        for i in range(env.num_envs):
            for p in paths:
                inc.AddTarget(Sdf.Path(p.replace("{ENV_REGEX_NS}", f"/World/envs/env_{i}")))
    arms.CreateFilteredGroupsRel().AddTarget(objs.GetPath())
