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
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SPEC = HERE / "spec.yaml"
SIDES = ("left", "right")
ARM_DOF = 6
GRASP_RADIUS_M = 0.03
BIG_JOINTS = (1, 2, 3)  # DM4340 joints; 4..6 are DM4310


def load_spec(path: Path = SPEC) -> dict:
    return yaml.safe_load(path.read_text())


def _quat_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def build_model(spec: dict, block_xy: list[list[float]]) -> mujoco.MjModel:
    """Compose the scene. ``block_xy`` are the (seeded) block centres in x, y."""
    xml = (HERE / spec["robot"]["model"]).read_text()
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
        for j in range(1, ARM_DOF + 1):
            big = j in BIG_JOINTS
            kp, kv = (sv["kp_big"], sv["kv_big"]) if big else (sv["kp_small"], sv["kv_small"])
            a = root.add_actuator(name=f"{side}_a{j}", target=f"{side}_joint{j}",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=kv)
        for finger in ("left_finger", "right_finger"):
            a = root.add_actuator(name=f"{side}_{finger}_a", target=f"{side}_{finger}",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=50.0, kv=2.0)
    # Welds, inactive until a grasp: box to the left gripper, each block to the right gripper.
    for obj, side in [("box", "left")] + [(f"block{i}", "right") for i in range(len(block_xy))]:
        root.add_equality(name=f"weld_{obj}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"{side}_link_6", name2=obj, active=False)
    return root.compile()


class World:
    """One episode. ``reset`` happens in the constructor; call ``step`` once per control period."""

    def __init__(self, spec: dict, seed: int):
        self.spec, self.seed = spec, seed
        rng = np.random.default_rng(seed)
        jit = spec["objects"]["blocks"]["jitter"]
        self.block_xy = [[x + rng.uniform(-jit, jit), y + rng.uniform(-jit, jit)]
                         for x, y, _ in spec["objects"]["blocks"]["at"]]
        h = spec["hand"]
        self.hand_target = np.array(h["target"], float) + [0, rng.uniform(-h["target_jitter_y"], h["target_jitter_y"]), 0]
        self.hand_delay = float(rng.uniform(*h["delay_after_first_release"]))
        self.m = build_model(spec, self.block_xy)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self._jq = {s: [m.jnt_qposadr[m.joint(f"{s}_joint{j}").id] for j in range(1, ARM_DOF + 1)] for s in SIDES}
        self._act = {s: [m.actuator(f"{s}_a{j}").id for j in range(1, ARM_DOF + 1)] for s in SIDES}
        self._fing = {s: [(m.actuator(f"{s}_{f}_a").id, m.joint(f"{s}_{f}").id) for f in ("left_finger", "right_finger")]
                      for s in SIDES}
        pts = spec["robot"]["points"]
        self.point_ids = [f"{s}_{p}" for s in SIDES for p in pts]
        self._pt = {}
        for s in SIDES:
            for p, ref in pts.items():
                self._pt[f"{s}_{p}"] = ("site", m.site(f"{s}_{ref['site']}").id) if "site" in ref \
                    else ("body", m.body(f"{s}_{ref['body']}").id)
        self.n_blocks = len(self.block_xy)
        for s in SIDES:
            q0 = spec["robot"]["start_q"][s]
            for a, v in zip(self._jq[s], q0):
                self.d.qpos[a] = v
            for a, v in zip(self._act[s], q0):
                self.d.ctrl[a] = v
            for a, j in self._fing[s]:  # fingers held open
                lo, hi = m.jnt_range[j]
                self.d.ctrl[a] = hi if m.joint(j).name.endswith("_left_finger") else lo
                self.d.qpos[m.jnt_qposadr[j]] = self.d.ctrl[a]
        mujoco.mj_forward(m, self.d)
        self.grip = {"left": 1.0, "right": 1.0}
        self.held = {"left": None, "right": None}
        self.first_release_t: float | None = None
        self.hand_phase = "away"
        self.hand_t0: float | None = None
        self.hand_done = False  # one visit per episode
        self.trace: list[dict] = []
        self._next_trace = 0.0
        self.control_dt = 1.0 / spec["robot"]["control_hz"]
        self._record()

    # ----- state ---------------------------------------------------------------------------------
    @property
    def t(self) -> float:
        return float(self.d.time)

    def q(self, side: str) -> np.ndarray:
        return np.array([self.d.qpos[a] for a in self._jq[side]])

    def point(self, pid: str) -> np.ndarray:
        kind, i = self._pt[pid]
        return (self.d.site_xpos[i] if kind == "site" else self.d.xpos[i]).copy()

    def hand(self) -> np.ndarray | None:
        return None if self.hand_phase == "away" else self.d.mocap_pos[0].copy()

    def human_distance(self) -> float | None:
        """What the proximity sensor reports: hand to the nearest named point of either arm."""
        h = self.hand()
        return None if h is None else min(float(np.linalg.norm(self.point(p) - h)) for p in self.point_ids)

    def obj_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.m.body(name).id].copy()

    def grip_point(self, name: str) -> np.ndarray:
        return self.d.site_xpos[self.m.site(f"{name}_grip").id].copy()

    def observation(self) -> dict:
        return {"t": self.t, "q": {s: self.q(s) for s in SIDES}, "grip": dict(self.grip),
                "human_distance_m": self.human_distance(), "state_age_ms": 0.0}

    # ----- control -------------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> None:
        """Apply one 14-D action for one control period."""
        action = np.asarray(action, float)
        assert action.shape == (14,), action.shape
        targets = {"left": action[0:6], "right": action[7:13]}
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
        h = self.spec["hand"]
        if self.hand_phase == "away":
            if self.hand_done:
                return
            start_at = (self.first_release_t + self.hand_delay) if self.first_release_t is not None else None
            if (start_at is not None and self.t >= start_at) or self.t >= h["fallback_at"]:
                self.hand_phase, self.hand_t0 = "approach", self.t
                self.d.mocap_pos[0] = h["start"]
            return
        start = np.array(h["start"], float)
        path = float(np.linalg.norm(self.hand_target - start))
        travel = path / h["speed"]
        el = self.t - self.hand_t0
        if el <= travel:
            self.hand_phase = "approach"
            pos = start + (self.hand_target - start) * (el / travel)
        elif el <= travel + h["hold"]:
            self.hand_phase = "hold"
            pos = self.hand_target
        elif el <= 2 * travel + h["hold"]:
            self.hand_phase = "leave"
            pos = self.hand_target + (start - self.hand_target) * ((el - travel - h["hold"]) / travel)
        else:
            self.hand_phase, self.hand_done = "away", True
            self.d.mocap_pos[0] = start
            return
        self.d.mocap_pos[0] = pos

    # ----- trace ---------------------------------------------------------------------------------
    def _record(self) -> None:
        h = self.hand()
        self.trace.append({
            "t": round(self.t, 6),
            "points": {p: self.point(p).tolist() for p in self.point_ids},
            "joints": {f"{s}/joint{j + 1}": float(v) for s in SIDES for j, v in enumerate(self.q(s))},
            "hand": None if h is None else h.tolist(),
            "objects": {n: self.obj_pos(n).tolist() for n in ["box"] + [f"block{i}" for i in range(self.n_blocks)]},
            "box_quat": self.d.xquat[self.m.body("box").id].tolist(),
            "grip": dict(self.grip),
            "arm_contacts": self._arm_contacts(),
        })
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
