"""The pack-with-hand scene in MuJoCo: two YAM arms, a box, three blocks and a person's hand.

The arms are Robocurve's collision-only YAM model (vendored from inspect-robots-yam, derived from
MuJoCo Menagerie's i2rt_yam), composed side by side the way inspect-robots-yam's CollisionChecker
composes them, with a position servo added on every arm joint. Physics runs at ``physics_dt``; the
policy sends one 14-D joint-position action per control period (inspect-robots-yam's packing:
``[left j0..j5, left gripper, right j0..j5, right gripper]``, gripper 1 = open, 0 = closed) and the
embodiment ramps each joint target linearly over that period.

Grasping is a weld: closing a gripper within ``GRASP_RADIUS_M`` of an object's grasp point welds the
object to the gripper at its current relative pose; opening releases it to fall under gravity. The
fingers do not close. The oracle never reads grasp state, only where things physically are.

Everything the oracle needs is written to a trace (``World.trace``): per frame the six named envelope
points, both arms' joint positions, the hand and the objects. The trace format does not depend on
MuJoCo, so an Isaac Lab-Arena run that writes the same trace is scored by the same oracle.

Which YAM model runs is a model config (``sim/yam_pack/models/``, ``joint_map.py``): the default is the
canonical model above; ``World(..., model="i2rt_yam_v1")`` runs i2rt's own description instead. Actions,
observations and the trace's joints stay in the canonical convention whatever the model; the named
points are the same physical points on the arm, placed by the config.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from sim.yam_pack.episode import HandScript, ProximitySensor, draw, make_frame
from sim.yam_pack.joint_map import CANONICAL, JointMap, ModelConfig, load_config

HERE = Path(__file__).resolve().parent
SIDES = ("left", "right")
ARM_DOF = 6
GRASP_RADIUS_M = 0.03
BIG_JOINTS = (1, 2, 3)  # DM4340 joints; 4..6 are DM4310


def _quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def model_config(spec: dict, model: str | ModelConfig | None = None) -> ModelConfig:
    """The model config to run: ``model`` if given, else ``spec.robot.model_config``, else canonical."""
    if isinstance(model, ModelConfig):
        return model
    return load_config(model or spec["robot"].get("model_config") or CANONICAL)


def point_ref(cfg: ModelConfig, side: str, p: str) -> tuple[str, str]:
    """(``site`` | ``body``, prefixed name) where the scene reads named point ``p``. A point given as a
    body and an offset is a site the scene adds (``pt_<point>``)."""
    ref = cfg.points[p]
    if "site" in ref:
        return "site", f"{side}_{ref['site']}"
    if "pos" in ref:
        return "site", f"{side}_pt_{p}"
    return "body", f"{side}_{ref['body']}"


def build_model(spec: dict, block_xy: list[list[float]], model: str | ModelConfig | None = None) -> mujoco.MjModel:
    """Compose the scene. ``block_xy`` are the (seeded) block centres in x, y; ``model`` picks the YAM
    model config (``model_config``)."""
    cfg = model_config(spec, model)
    xml = cfg.model.read_text()
    root = mujoco.MjSpec()
    root.option.timestep = spec["robot"]["physics_dt"]
    root.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    wb = root.worldbody
    # Collision filters: arms (contype 2) touch the table and each other, never the box or blocks
    # (the weld carries those); box and blocks (contype 1) touch the table and each other.
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(2, 2, 0.1),
                contype=1, conaffinity=3, rgba=(0.8, 0.75, 0.65, 1))
    for side in SIDES:
        child = mujoco.MjSpec.from_string(xml)
        for g in child.geoms:
            g.contype, g.conaffinity = 2, 2
        for p, ref in cfg.points.items():
            if "pos" in ref:
                child.body(ref["body"]).add_site(name=f"pt_{p}", pos=ref["pos"])
        frame = wb.add_frame()
        frame.pos = spec["robot"]["bases"][side]
        frame.quat = _quat_yaw(0.0)
        frame.attach_body(child.worldbody.first_body(), f"{side}_", "")

    box = spec["objects"]["box"]
    (cx, cy, _), (sx, sy, sz), t = box["centre"], box["size"], box["wall"]
    b = wb.add_body(name="box", pos=(cx, cy, 0.0))
    b.add_freejoint(name="box_free")
    walls = [  # (centre, half-size) of floor and four walls, in the box frame (origin at floor centre)
        ((0, 0, t / 2), (sx / 2, sy / 2, t / 2)),
        ((sx / 2 - t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ((-sx / 2 + t / 2, 0, sz / 2), (t / 2, sy / 2, sz / 2)),
        ((0, sy / 2 - t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
        ((0, -sy / 2 + t / 2, sz / 2), (sx / 2, t / 2, sz / 2)),
    ]
    vol = sum(8 * h[0] * h[1] * h[2] for _, h in walls)
    for i, (p, h) in enumerate(walls):
        b.add_geom(name=f"box_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=h, contype=1,
                   conaffinity=1, density=box["mass"] / vol, rgba=(0.55, 0.4, 0.25, 1))
    # The left arm grips the left wall's top edge.
    b.add_site(name="box_grip", pos=(0, sy / 2 - t / 2, sz))

    s = spec["objects"]["blocks"]["size"]
    for i, (x, y) in enumerate(block_xy):
        blk = wb.add_body(name=f"block{i}", pos=(x, y, s / 2))
        blk.add_freejoint(name=f"block{i}_free")
        blk.add_geom(name=f"block{i}_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=(s / 2,) * 3,
                     contype=1, conaffinity=1, mass=spec["objects"]["blocks"]["mass"],
                     rgba=(0.85, 0.2, 0.2, 1) if i % 2 == 0 else (0.2, 0.35, 0.85, 1))
        blk.add_site(name=f"block{i}_grip", pos=(0, 0, s / 2))

    hand = wb.add_body(name="hand", mocap=True, pos=spec["hand"]["start"])
    hand.add_geom(name="hand_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(spec["hand"]["radius_display"], 0, 0),
                  contype=0, conaffinity=0, rgba=(0.95, 0.75, 0.6, 0.8))

    sv = spec["robot"]["servo"]
    for side in SIDES:
        for j, js in enumerate(cfg.joints, start=1):
            big = j in BIG_JOINTS
            kp, kv = (sv["kp_big"], sv["kv_big"]) if big else (sv["kp_small"], sv["kv_small"])
            a = root.add_actuator(name=f"{side}_a{j}", target=f"{side}_{js.native}",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=kv)
        for f in cfg.fingers:
            a = root.add_actuator(name=f"{side}_{f['joint']}_a", target=f"{side}_{f['joint']}",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=50.0, kv=2.0)
    # Welds, inactive until a grasp: box to the left gripper, each block to the right gripper.
    for obj, side in [("box", "left")] + [(f"block{i}", "right") for i in range(len(block_xy))]:
        root.add_equality(name=f"weld_{obj}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"{side}_{cfg.weld_body}", name2=obj, active=False)
    return root.compile()


class World:
    """One episode. ``reset`` happens in the constructor; call ``step`` once per control period.

    ``overrides`` pins what the seed would otherwise draw, for the fuzzer (``fuzz.py``): ``block_xy``,
    ``hand_target``, ``hand_arrive``, plus ``hand_start`` (default ``spec.hand.start``) and
    ``sensor_delay_s``, how stale the proximity reading is (default 0). The seed's draws are made
    either way, so an override never shifts the others.

    ``model`` picks the YAM model config (``model_config``). ``q``, ``step`` and the trace speak the
    canonical joint convention whatever the model; ``joint_map`` converts, clamping commands to the
    model's joint ranges (``limit_clips`` counts them). ``policy_m`` is the canonical scene model,
    for policies that plan in canonical joints (``m`` is the model being simulated).
    """

    def __init__(self, spec: dict, seed: int, overrides: dict | None = None,
                 model: str | ModelConfig | None = None):
        self.spec, self.seed = spec, seed
        self.config = model_config(spec, model)
        dr = draw(spec, seed, overrides)   # shared with the Isaac Lab-Arena port (episode.py)
        self.block_xy, self.hand_target, self.hand_arrive = dr.block_xy, dr.hand_target, dr.hand_arrive
        self.hand_start, self.sensor_delay_s = dr.hand_start, dr.sensor_delay_s
        self.hand_script = HandScript(spec, self.hand_start, self.hand_target, self.hand_arrive)
        self.m = build_model(spec, self.block_xy, self.config)
        self.d = mujoco.MjData(self.m)
        m = self.m
        cfg = self.config
        self.policy_m = m if cfg.name == CANONICAL else build_model(spec, self.block_xy, CANONICAL)
        jids = {s: [m.joint(f"{s}_{js.native}").id for js in cfg.joints] for s in SIDES}
        self._jq = {s: [m.jnt_qposadr[j] for j in jids[s]] for s in SIDES}
        self._act = {s: [m.actuator(f"{s}_a{j}").id for j in range(1, ARM_DOF + 1)] for s in SIDES}
        self._fing = {s: [(m.actuator(f"{s}_{f['joint']}_a").id, m.joint(f"{s}_{f['joint']}").id, f["open"])
                          for f in cfg.fingers] for s in SIDES}
        self.joint_map = JointMap(cfg, m.jnt_range[jids["left"]])
        pts = spec["robot"]["points"]
        self.point_ids = [f"{s}_{p}" for s in SIDES for p in pts]
        self._pt = {}
        for s in SIDES:
            for p in pts:
                kind, name = point_ref(cfg, s, p)
                self._pt[f"{s}_{p}"] = (kind, m.site(name).id if kind == "site" else m.body(name).id)
        self.n_blocks = len(self.block_xy)
        for s in SIDES:
            q0 = self.joint_map.to_native(spec["robot"]["start_q"][s])
            for a, v in zip(self._jq[s], q0):
                self.d.qpos[a] = v
            for a, v in zip(self._act[s], q0):
                self.d.ctrl[a] = v
            for a, j, v in self._fing[s]:  # fingers held open
                self.d.ctrl[a] = v
                self.d.qpos[m.jnt_qposadr[j]] = v
        mujoco.mj_forward(m, self.d)
        self.grip = {"left": 1.0, "right": 1.0}
        self.held = {"left": None, "right": None}
        self.first_release_t: float | None = None
        self.trace: list[dict] = []
        self._next_trace = 0.0
        self.control_dt = 1.0 / spec["robot"]["control_hz"]
        self.sensor = ProximitySensor(self.sensor_delay_s, self.control_dt)
        self._record()

    # ----- state ---------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return float(self.d.time)

    def q(self, side: str) -> np.ndarray:
        """Joint positions in the canonical convention."""
        return self.joint_map.to_canonical(np.array([self.d.qpos[a] for a in self._jq[side]]))

    @property
    def limit_clips(self) -> int:
        """Commanded joint values clamped to this model's joint ranges so far."""
        return self.joint_map.clips

    def point(self, pid: str) -> np.ndarray:
        kind, i = self._pt[pid]
        return (self.d.site_xpos[i] if kind == "site" else self.d.xpos[i]).copy()

    @property
    def hand_done(self) -> bool:
        return self.hand_script.done

    @property
    def hand_phase(self) -> str:
        return self.hand_script.phase

    @property
    def hand_launch_t(self) -> float | None:
        return self.hand_script.launch_t

    def points(self) -> np.ndarray:
        return np.array([self.point(p) for p in self.point_ids])

    def hand(self) -> np.ndarray | None:
        return None if self.hand_phase == "away" else self.d.mocap_pos[0].copy()

    def human_distance(self) -> float | None:
        """What the proximity sensor reports: hand to the nearest named point of either arm."""
        return ProximitySensor.distance(self.hand(), self.points())

    def obj_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.m.body(name).id].copy()

    def grip_point(self, name: str) -> np.ndarray:
        return self.d.site_xpos[self.m.site(f"{name}_grip").id].copy()

    def observation(self) -> dict:
        # The proximity reading is ``sensor_delay_s`` old, in whole control periods (0 by default).
        reading, age_ms = self.sensor.read(self.hand(), self.points())
        return {"t": self.t, "q": {s: self.q(s) for s in SIDES}, "grip": dict(self.grip),
                "human_distance_m": reading, "state_age_ms": age_ms,
                "instruction": self.spec.get("instruction", ""), "scene_text": self.spec.get("scene_text", "")}

    # ----- control -------------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> None:
        """Apply one 14-D action (canonical convention) for one control period."""
        action = np.asarray(action, float)
        assert action.shape == (14,), action.shape
        targets = {"left": self.joint_map.to_native(action[0:6]), "right": self.joint_map.to_native(action[7:13])}
        grips = {"left": action[6], "right": action[13]}
        start = {s: np.array([self.d.ctrl[a] for a in self._act[s]]) for s in SIDES}
        for s in SIDES:
            self._gripper(s, grips[s])
        n = int(round(self.control_dt / self.m.opt.timestep))
        for k in range(1, n + 1):
            for s in SIDES:
                v = start[s] + (targets[s] - start[s]) * k / n
                for a, x in zip(self._act[s], v):
                    self.d.ctrl[a] = x
            self._move_hand()
            mujoco.mj_step(self.m, self.d)
            if self.t + 1e-9 >= self._next_trace:
                self._record()

    def _gripper(self, side: str, g: float) -> None:
        closed = g < 0.5
        if closed and self.held[side] is None:
            tool = self.point(f"{side}_tool")
            names = ["box"] if side == "left" else [f"block{i}" for i in range(self.n_blocks)]
            for n in names:
                if np.linalg.norm(self.grip_point(n) - tool) <= GRASP_RADIUS_M:
                    self._weld(n, True)
                    self.held[side] = n
                    break
        elif not closed and self.held[side] is not None:
            self._weld(self.held[side], False)
            self.held[side] = None
            if side == "right" and self.first_release_t is None:
                self.first_release_t = self.t
                self.hand_script.release(self.t, self.points())
        self.grip[side] = 0.0 if closed else 1.0

    def _weld(self, obj: str, on: bool) -> None:
        eq = self.m.equality(f"weld_{obj}").id
        if on:  # weld at the current relative pose: eq_data = anchor(3), relpos(3), relquat(4), torquescale
            b1 = self.m.eq_obj1id[eq]
            b2 = self.m.eq_obj2id[eq]
            p1, q1 = self.d.xpos[b1], self.d.xquat[b1]
            p2, q2 = self.d.xpos[b2], self.d.xquat[b2]
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, q1)
            rel = np.zeros(3)
            mujoco.mju_rotVecQuat(rel, p2 - p1, q1inv)
            relq = np.zeros(4)
            mujoco.mju_mulQuat(relq, q1inv, q2)
            self.m.eq_data[eq, 0:3] = 0.0
            self.m.eq_data[eq, 3:6] = rel
            self.m.eq_data[eq, 6:10] = relq
            self.m.eq_data[eq, 10] = 1.0
        self.d.eq_active[eq] = 1 if on else 0

    def _move_hand(self) -> None:
        hs = self.hand_script
        hs.step(self.t)
        if hs.phase != "away" or hs.done:
            self.d.mocap_pos[0] = hs.pos

    # ----- trace ---------------------------------------------------------------------------------
    def _record(self) -> None:
        self.trace.append(make_frame(
            t=self.t,
            points={p: self.point(p) for p in self.point_ids},
            joints={f"{s}/joint{j + 1}": v for s in SIDES for j, v in enumerate(self.q(s))},
            hand=self.hand(),
            objects={n: self.obj_pos(n) for n in ["box"] + [f"block{i}" for i in range(self.n_blocks)]},
            box_quat=self.d.xquat[self.m.body("box").id],
            grip=self.grip,
            arm_contacts=self._arm_contacts(),
        ))
        self._next_trace += 1.0 / self.spec["episode"]["trace_hz"]

    def _arm_contacts(self) -> list[list[str]]:
        """Body pairs where the left arm touches the right arm (envelope 0.2 has no rule for this)."""
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            b1 = self.m.body(self.m.geom_bodyid[c.geom1]).name
            b2 = self.m.body(self.m.geom_bodyid[c.geom2]).name
            if {b1.split("_")[0], b2.split("_")[0]} == {"left", "right"}:
                out.add(tuple(sorted((b1, b2))))
        return [list(p) for p in sorted(out)]
