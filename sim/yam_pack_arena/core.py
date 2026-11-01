"""The pack-with-hand episode, driven through any simulator that can do a handful of things.

``PackDriver`` is ``sim/yam_pack/scene.World``'s control and trace logic with the simulator calls
pulled out behind ``Backend``: one control period takes a 14-D action, ramps the joint targets
linearly over its physics substeps, moves the hand on ``HandScript``'s schedule, carries held
objects, and writes trace frames at ``trace_hz`` in the format the oracle reads.

The one deliberate change from ``World`` is the grasp. MuJoCo welds a grasped object to ``link_6``
with an equality constraint; Isaac Lab has no runtime weld, so here a grasp records the object's pose
relative to ``link_6`` and, before every physics step, pushes the object toward that pose with a
force and torque at its centre of mass (a soft weld): a spring-damper on the pose and velocity error
with the stiffness and damping of MuJoCo's default weld (solref 0.02 s, damping ratio 1), plus the
object's weight. The object stays a dynamic body, so what rests in it rides along by friction.

An earlier version wrote the object's pose and velocity straight into the simulator each substep (a
kinematic carry). The pose was right, but a box moved that way slid out from under its contents: the
contact solve never saw the box move, so friction never dragged the blocks, and a block the weld
carries 6 cm stayed where it landed. That is what made the tasks' in-box rule look too strict under
the driver (``sim/yam_pack_tasks/README.md``); the rule is task 1's again now.

``MujocoBackend`` below runs this same driver in MuJoCo with the welds left off, which is how the
carry is checked on a CPU: the reference policies must get the verdicts they get with welds
(``tests/test_sim_yam_arena.py``, ``carry_check.py``).

Order of calls per physics substep, the same in both simulators:

    driver.begin(action)                  # once per control period: grasp/release, ramp start
    for k in substeps:
        driver.pre_physics()              # targets, hand, carried objects
        <one physics step>
        driver.post_physics()             # clock, trace frame when due

Isaac Lab's env loop has no hook after its last substep, so the Isaac ``ActionTerm`` calls
``post_physics`` at the start of the next ``apply_actions`` and the observation term calls
``flush()`` before reading; both are no-ops when nothing is pending.
"""
from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

from sim.yam_pack.episode import SIDES, HandScript, ProximitySensor, draw, make_frame

GRASP_RADIUS_M = 0.03    # same as scene.GRASP_RADIUS_M
ARM_DOF = 6
# The soft weld, per unit mass (and per unit inertia for rotation): MuJoCo's default constraint
# (solref 0.02 s, damping ratio 1, dmax 0.95) gives b = 2 / (dmax * 0.02) and k = 1 / (dmax * 0.02)^2.
CARRY_K = 1.0 / (0.95 * 0.02) ** 2     # 1/s^2, about 2770
CARRY_B = 2.0 / (0.95 * 0.02)          # 1/s, about 105


class Backend(Protocol):
    """What the driver needs from a simulator. Positions in the table frame (metres), quaternions
    (w, x, y, z), joint values in radians, sides "left"/"right", objects "box", "block0".."""

    physics_dt: float

    def joint_q(self, side: str) -> np.ndarray: ...
    def set_arm_target(self, side: str, q: np.ndarray) -> None: ...
    def point(self, pid: str) -> np.ndarray: ...                      # "left_tool", "right_elbow", ...
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]: ...
    def grip_point(self, name: str) -> np.ndarray: ...
    def link6_state(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    #   position, quaternion, linear velocity at the link origin, angular velocity (world frame)
    def set_object_state(self, name: str, pos, quat, linvel, angvel) -> None: ...
    def object_state(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    #   origin position, quaternion, centre of mass, linear velocity at the centre of mass, angular velocity
    def drive_object(self, name: str, lin_acc, ang_acc) -> None: ...
    #   hold the force and torque at the centre of mass that give these accelerations (world frame)
    #   against gravity, until the next call; None clears it
    def set_hand(self, pos: np.ndarray) -> None: ...
    def arm_contacts(self) -> list[list[str]]: ...


# ----- quaternion helpers (w, x, y, z) ----------------------------------------------------------
def qmul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def qinv(a) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(a, float))
    return out


def qrot(a, v) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(a, float))
    return out


class PackDriver:
    """``joint_map`` (``sim.yam_pack.joint_map.JointMap``) converts between the canonical joints the
    policies, the trace and ``ctrl`` speak and the backend's own model; None means the backend's
    model is the canonical one. Commands are clamped to the map's ranges once per action."""

    def __init__(self, backend: Backend, spec: dict, seed: int, overrides: dict | None = None,
                 draws=None, joint_map=None):
        self.b, self.spec, self.seed = backend, spec, seed
        self.joint_map = joint_map
        dr = draws or draw(spec, seed, overrides)
        self.block_xy, self.hand_target, self.hand_arrive = dr.block_xy, dr.hand_target, dr.hand_arrive
        self.hand_start = dr.hand_start
        self.hand_script = HandScript(spec, dr.hand_start, dr.hand_target, dr.hand_arrive)
        self.control_dt = 1.0 / spec["robot"]["control_hz"]
        self.n_sub = int(round(self.control_dt / backend.physics_dt))
        self.sensor = ProximitySensor(dr.sensor_delay_s, self.control_dt)
        self.point_ids = [f"{s}_{p}" for s in SIDES for p in spec["robot"]["points"]]
        self.objects = ["box"] + [f"block{i}" for i in range(len(self.block_xy))]
        self.ctrl = {s: np.array(spec["robot"]["start_q"][s], float) for s in SIDES}
        self.grip = {"left": 1.0, "right": 1.0}
        self.held: dict[str, str | None] = {"left": None, "right": None}
        self.rel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.first_release_t: float | None = None
        self.t = 0.0
        self.trace: list[dict] = []
        self._next_trace = 0.0
        self._pending = False
        self._k = 0
        self._start = dict(self.ctrl)
        self._target = dict(self.ctrl)
        self._record()

    # ----- state -----------------------------------------------------------------------------------
    @property
    def hand_done(self) -> bool:
        return self.hand_script.done

    def points(self) -> np.ndarray:
        return np.array([self.b.point(p) for p in self.point_ids])

    def hand(self) -> np.ndarray | None:
        return self.hand_script.position()

    def q(self, side: str) -> np.ndarray:
        q = np.asarray(self.b.joint_q(side), float)
        return q if self.joint_map is None else self.joint_map.to_canonical(q)

    def observation(self) -> dict:
        self.flush()
        reading, age_ms = self.sensor.read(self.hand(), self.points())
        return {"t": self.t, "q": {s: self.q(s) for s in SIDES}, "grip": dict(self.grip),
                "human_distance_m": reading, "state_age_ms": age_ms,
                "instruction": self.spec.get("instruction", ""), "scene_text": self.spec.get("scene_text", "")}

    # ----- control ---------------------------------------------------------------------------------
    def begin(self, action) -> None:
        self.flush()
        a = np.asarray(action, float)
        assert a.shape == (14,), a.shape
        self._start = {s: self.ctrl[s].copy() for s in SIDES}
        self._target = {"left": a[0:6].copy(), "right": a[7:13].copy()}
        if self.joint_map is not None:   # clamp to the model's ranges, in canonical terms
            jm = self.joint_map
            self._target = {s: jm.to_canonical(jm.to_native(v)) for s, v in self._target.items()}
        for s, g in (("left", a[6]), ("right", a[13])):
            self._gripper(s, g)
        self._k = 0

    def pre_physics(self) -> None:
        self.flush()
        self._k = min(self._k + 1, self.n_sub)
        for s in SIDES:
            self.ctrl[s] = self._start[s] + (self._target[s] - self._start[s]) * self._k / self.n_sub
            self.b.set_arm_target(s, self.ctrl[s] if self.joint_map is None
                                  else self.joint_map.to_native(self.ctrl[s], clamp=False))
        hs = self.hand_script
        hs.step(self.t)
        if hs.phase != "away" or hs.done:
            self.b.set_hand(hs.pos)
        self._carry()

    def post_physics(self) -> None:
        self.t += self.b.physics_dt
        self._pending = True
        self.flush()

    def flush(self) -> None:
        """Write the trace frame the last physics step made due, if any."""
        if not self._pending:
            return
        self._pending = False
        if self.t + 1e-9 >= self._next_trace:
            self._record()

    def step(self, action, physics_step) -> None:
        """One control period with ``physics_step()`` advancing the simulator by one substep."""
        self.begin(action)
        for _ in range(self.n_sub):
            self.pre_physics()
            physics_step()
            self.post_physics()

    def _gripper(self, side: str, g: float) -> None:
        closed = g < 0.5
        if closed and self.held[side] is None:
            tool = self.b.point(f"{side}_tool")
            names = ["box"] if side == "left" else [o for o in self.objects if o != "box"]
            for n in names:
                if np.linalg.norm(self.b.grip_point(n) - tool) <= GRASP_RADIUS_M:
                    self._attach(side, n)
                    break
        elif not closed and self.held[side] is not None:
            self.b.drive_object(self.held[side], None, None)
            self.held[side] = None
            if side == "right" and self.first_release_t is None:
                self.first_release_t = self.t
                self.hand_script.release(self.t, self.points())
        self.grip[side] = 0.0 if closed else 1.0

    def _attach(self, side: str, obj: str) -> None:
        lp, lq, _, _ = self.b.link6_state(side)
        op, oq = self.b.object_pose(obj)
        lqi = qinv(lq)
        self.rel[obj] = (qrot(lqi, op - lp), qmul(lqi, oq))
        self.held[side] = obj

    def _carry(self) -> None:
        """The soft weld: for each held object, the acceleration that takes its centre of mass and
        orientation to where the grasp put them relative to ``link_6``, at the link's velocity there."""
        for side, obj in self.held.items():
            if obj is None:
                continue
            lp, lq, lv, lw = self.b.link6_state(side)
            rp, rq = self.rel[obj]
            p, q, com, v, w = self.b.object_state(obj)
            tq = qmul(lq, rq)
            tcom = lp + qrot(lq, rp) + qrot(tq, qrot(qinv(q), com - p))   # where the centre of mass belongs
            tv = lv + np.cross(lw, tcom - lp)
            qe = qmul(tq, qinv(q))
            if qe[0] < 0:
                qe = -qe
            rot = np.zeros(3)
            mujoco.mju_quat2Vel(rot, qe, 1.0)                            # rotation vector, object to target
            self.b.drive_object(obj, CARRY_K * (tcom - com) + CARRY_B * (tv - v), CARRY_K * rot + CARRY_B * (lw - w))

    # ----- trace -----------------------------------------------------------------------------------
    def _record(self) -> None:
        box_p, box_q = self.b.object_pose("box")
        self.trace.append(make_frame(
            t=self.t,
            points={p: self.b.point(p) for p in self.point_ids},
            joints={f"{s}/joint{j + 1}": v for s in SIDES for j, v in enumerate(self.q(s))},
            hand=self.hand(),
            objects={n: (box_p if n == "box" else self.b.object_pose(n)[0]) for n in self.objects},
            box_quat=box_q,
            grip=self.grip,
            arm_contacts=self.b.arm_contacts(),
        ))
        self._next_trace += 1.0 / self.spec["episode"]["trace_hz"]


# ----- MuJoCo backend: the same scene, welds off, soft-weld carry -----------------------------
class MujocoBackend:
    """``sim/yam_pack``'s MuJoCo scene behind the ``Backend`` protocol, with the welds never switched
    on. Used to check the carry and the driver on a CPU; the shipped MuJoCo results stay on ``World``."""

    def __init__(self, spec: dict, block_xy: list[list[float]], model=None):
        from sim.yam_pack import scene
        from sim.yam_pack.joint_map import JointMap

        self.scene = scene
        self.config = cfg = scene.model_config(spec, model)
        self.m = scene.build_model(spec, block_xy, cfg)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.physics_dt = float(m.opt.timestep)
        jids = {s: [m.joint(f"{s}_{js.native}").id for js in cfg.joints] for s in SIDES}
        self._jq = {s: [m.jnt_qposadr[j] for j in jids[s]] for s in SIDES}
        self._act = {s: [m.actuator(f"{s}_a{j}").id for j in range(1, ARM_DOF + 1)] for s in SIDES}
        self.joint_map = JointMap(cfg, m.jnt_range[jids["left"]])
        self._pt = {}
        for s in SIDES:
            for p in spec["robot"]["points"]:
                kind, name = scene.point_ref(cfg, s, p)
                self._pt[f"{s}_{p}"] = (kind, m.site(name).id if kind == "site" else m.body(name).id)
        for s in SIDES:
            q0 = self.joint_map.to_native(spec["robot"]["start_q"][s])
            for a, v in zip(self._jq[s], q0):
                self.d.qpos[a] = v
            for a, v in zip(self._act[s], q0):
                self.d.ctrl[a] = v
            for f in cfg.fingers:
                a, j = m.actuator(f"{s}_{f['joint']}_a").id, m.joint(f"{s}_{f['joint']}").id
                self.d.ctrl[a] = f["open"]
                self.d.qpos[m.jnt_qposadr[j]] = f["open"]
        mujoco.mj_forward(m, self.d)

    def physics_step(self) -> None:
        mujoco.mj_step(self.m, self.d)

    def joint_q(self, side):
        return np.array([self.d.qpos[a] for a in self._jq[side]])

    def set_arm_target(self, side, q):
        for a, x in zip(self._act[side], q):
            self.d.ctrl[a] = x

    def point(self, pid):
        kind, i = self._pt[pid]
        return (self.d.site_xpos[i] if kind == "site" else self.d.xpos[i]).copy()

    def object_pose(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def grip_point(self, name):
        return self.d.site_xpos[self.m.site(f"{name}_grip").id].copy()

    def link6_state(self, side):
        b = self.m.body(f"{side}_{self.config.weld_body}").id   # link_6 on the canonical model
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        w, lin_com = v[:3], v[3:]
        # mj_objectVelocity reports the linear velocity at the body's centre of mass; move it to the origin.
        lin = lin_com + np.cross(w, self.d.xpos[b] - self.d.xipos[b])
        return self.d.xpos[b].copy(), self.d.xquat[b].copy(), lin, w.copy()

    def set_object_state(self, name, pos, quat, linvel, angvel):
        j = self.m.body(name).jntadr[0]
        qa, va = self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]
        self.d.qpos[qa:qa + 3] = pos
        self.d.qpos[qa + 3:qa + 7] = quat
        # Free-joint velocity: linear in world frame, angular in the body's local frame.
        self.d.qvel[va:va + 3] = linvel
        self.d.qvel[va + 3:va + 6] = qrot(qinv(quat), angvel)

    def object_state(self, name):
        b = self.m.body(name).id
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        return self.d.xpos[b].copy(), self.d.xquat[b].copy(), self.d.xipos[b].copy(), v[3:].copy(), v[:3].copy()

    def drive_object(self, name, lin_acc, ang_acc):
        b = self.m.body(name).id
        if lin_acc is None:
            self.d.xfrc_applied[b] = 0.0
            return
        R = self.d.ximat[b].reshape(3, 3)
        inertia = R @ np.diag(self.m.body_inertia[b]) @ R.T
        w = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, w, 0)
        # xfrc_applied is a world-frame force and torque at the body's centre of mass.
        self.d.xfrc_applied[b, :3] = self.m.body_mass[b] * (np.asarray(lin_acc) - self.m.opt.gravity)
        self.d.xfrc_applied[b, 3:] = inertia @ np.asarray(ang_acc) + np.cross(w[:3], inertia @ w[:3])

    def set_hand(self, pos):
        self.d.mocap_pos[0] = pos

    def arm_contacts(self):
        out = set()
        for c in self.d.contact[: self.d.ncon]:
            b1 = self.m.body(self.m.geom_bodyid[c.geom1]).name
            b2 = self.m.body(self.m.geom_bodyid[c.geom2]).name
            if {b1.split("_")[0], b2.split("_")[0]} == {"left", "right"}:
                out.add(tuple(sorted((b1, b2))))
        return [list(p) for p in sorted(out)]


class PolicyView:
    """What ``sim.yam_pack.policies.Packer`` reads from a world (``m``, ``q``, ``block_xy``, ``spec``),
    built from a driver, so the reference policies run unchanged against any backend. ``m`` is the
    MuJoCo model the policies' IK uses; with the Isaac backend it is the same compiled MJCF, used for
    IK only, never stepped."""

    def __init__(self, driver: PackDriver, model: mujoco.MjModel):
        self.driver, self.m, self.spec, self.block_xy = driver, model, driver.spec, driver.block_xy

    def q(self, side: str) -> np.ndarray:
        return self.driver.q(side)


def episode(policy: str, seed: int, spec: dict, overrides: dict | None = None,
            model=None) -> tuple[list[dict], PackDriver]:
    """``sim.yam_pack.run.episode`` through the driver and the MuJoCo backend (soft-weld carry).
    ``model`` is a YAM model config (``sim.yam_pack.joint_map``); the policies plan on the canonical
    model and the driver's joint map carries their joints onto this one."""
    from sim.yam_pack.joint_map import CANONICAL
    from sim.yam_pack.policies import Packer

    dr = draw(spec, seed, overrides)
    be = MujocoBackend(spec, dr.block_xy, model)
    canonical = be.config.name == CANONICAL
    drv = PackDriver(be, spec, seed, draws=dr, joint_map=None if canonical else be.joint_map)
    pol = Packer(PolicyView(drv, be.m if canonical else be.scene.build_model(spec, dr.block_xy, CANONICAL)), policy)
    settle, end_at = 1.0, None
    while drv.t < spec["episode"]["max_time"]:
        drv.step(pol.act(drv.observation()), be.physics_step)
        if end_at is None and pol.done() and drv.hand_done:
            end_at = drv.t + settle
        if end_at is not None and drv.t >= end_at:
            break
    drv.flush()
    return drv.trace, drv
