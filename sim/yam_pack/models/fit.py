"""Derive and check a model config's joint map against the canonical YAM, from the model files alone.

    python -m sim.yam_pack.models.fit --check i2rt_yam_v1          # residuals of a committed config
    python -m sim.yam_pack.models.fit --fit path/to/new_yam.xml \
        --joints joint1,joint2,joint3,joint4,joint5,joint6 --bodies link3,link5,gripper --finger joint7

Both models are compared one arm at a time, base at the origin, through MuJoCo forward kinematics.

``fit`` finds each joint's sign and zero offset (``native = sign * canonical + offset``) from the
base out: the sign from the two models' axis directions, the offset from the angle between something
the joint turns (the next axis, the next axis's line, or for the tool roll the finger's slide axis),
averaged over random poses; the spread of those angles is the fit's error. It then places the envelope's
named points: for each, the point fixed to the given native body that best tracks the canonical
point, solved together with one translation of the whole arm (``lift``), which is how far the two
models disagree about where the arm sits on its base. The lift is reported, never applied: the scene
puts each model's arm where that model says it is.

``check`` samples poses inside both models' joint ranges and reports, under the config's map: the
angle between corresponding joint axes, the distance between corresponding axis lines, and the
distance between corresponding named points, with and without the lift. Zero axis angles mean the map
is exact up to the lift; anything else is a geometric difference no joint map can remove.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.yam_pack.joint_map import CANONICAL, JointMap, ModelConfig, load_config


class Arm:
    """One arm model on its own, for kinematics."""

    def __init__(self, path: Path, joints: list[str]):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.jid = [self.m.joint(j).id for j in joints]
        self.qadr = [self.m.jnt_qposadr[j] for j in self.jid]
        self.ranges = self.m.jnt_range[self.jid].copy()

    def set(self, q: np.ndarray) -> None:
        self.d.qpos[:] = 0.0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.xaxis[self.jid].copy(), self.d.xanchor[self.jid].copy()

    def joint_axis(self, name: str) -> np.ndarray:
        return self.d.xaxis[self.m.joint(name).id].copy()

    def body(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def point(self, ref: dict) -> np.ndarray:
        if "site" in ref:
            return self.d.site_xpos[self.m.site(ref["site"]).id].copy()
        p, R = self.body(ref["body"])
        return p + R @ np.asarray(ref.get("pos", (0.0, 0.0, 0.0)), float)


def _arm(cfg: ModelConfig) -> Arm:
    return Arm(cfg.model, [j.native for j in cfg.joints])


def _poses(ranges: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(ranges[:, 0], ranges[:, 1], size=(n, len(ranges)))


def fit(model: Path, joints: list[str], bodies: dict[str, str], finger: str,
        canonical: ModelConfig | None = None, n: int = 60, seed: int = 0) -> dict:
    """A config's joint and point numbers for ``model`` (see the module docstring)."""
    canon = canonical or load_config(CANONICAL)
    ca, na = _arm(canon), Arm(model, joints)
    cfinger = canon.fingers[0]["joint"]
    qs = _poses(ca.ranges, n, seed)

    def feature(arm: Arm, k: int, fj: str) -> tuple[np.ndarray, bool]:
        """Something joint k turns that depends on joints 1..k only, seen across axis k, and whether
        it is a bare axis direction (whose sign is a naming convention, so it fixes the offset only up
        to pi). In order: the common perpendicular from axis k's line to the next axis's line; the next
        axis's direction; for the last joint, the finger's slide axis. A lift along joint 1 cancels."""
        ax, an = arm.axes()
        if k == 5:
            v = arm.joint_axis(fj)
            return (v - (v @ ax[k]) * ax[k]) / np.linalg.norm(v - (v @ ax[k]) * ax[k]), True
        u, v, w0 = ax[k], ax[k + 1], an[k + 1] - an[k]
        b = float(u @ v)
        if abs(b) > 1 - 1e-9:                      # parallel lines: their offset
            w = w0 - (w0 @ u) * u
        else:                                      # skew or crossing: the common perpendicular
            t = (w0 @ u - b * (w0 @ v)) / (1 - b * b)
            s_ = (b * (w0 @ u) - w0 @ v) / (1 - b * b)
            w = (an[k + 1] + s_ * v) - (an[k] + t * u)
        if np.linalg.norm(w) > 1e-4:
            return w / np.linalg.norm(w), False
        v = v - (v @ u) * u
        return v / np.linalg.norm(v), True

    def solve(flip: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, float, list[bool]]:
        """Joint by joint from the base: the sign from the axis directions, the offset from the angle
        (about the native axis) between the two models' feature, with the native joint at zero.
        ``flip[k]`` adds pi where the feature is a bare direction."""
        signs, offs, spread, ambiguous = np.ones(6), np.zeros(6), 0.0, []
        for k in range(6):
            ks, kc, amb = [], [], False
            for q in qs:
                qc = q.copy()
                qc[k:] = 0.0
                ca.set(qc)
                qn = signs * qc + offs
                qn[k:] = 0.0
                na.set(qn)
                axn = na.axes()[0][k]
                ks.append(float(ca.axes()[0][k] @ axn))
                (fc, a1), (fn, a2) = feature(ca, k, cfinger), feature(na, k, finger)
                amb |= a1 or a2
                kc.append(np.arctan2(axn @ np.cross(fn, fc), fn @ fc))
            if min(abs(v) for v in ks) < 1 - 1e-6:
                raise ValueError(f"joint {k + 1}: axes are not parallel in the two models, so no joint map fits")
            ambiguous.append(amb)
            z = np.exp(1j * np.array(kc))
            mean = z.mean() / abs(z.mean())
            offs[k] = float(np.angle(-mean if amb and flip[k] else mean))
            spread = max(spread, float(np.max(np.abs(np.angle(z / mean)))))
            signs[k] = np.sign(np.mean(ks))
        return signs, offs, spread, ambiguous

    def tool_residual(signs, offs) -> float:
        """RMS of the best rigid placement of the canonical tool point on the native weld body."""
        A, b = [], []
        for q in qs:
            ca.set(q)
            na.set(signs * q + offs)
            x, R = na.body(bodies["tool"])
            A.append(np.hstack([R, np.eye(3)]))
            b.append(ca.point(canon.points["tool"]) - x)
        A, b = np.vstack(A), np.concatenate(b)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(np.sqrt(np.mean((A @ sol - b) ** 2)))

    _, _, _, amb = solve((0,) * 6)
    best = None
    for bits in range(2 ** sum(amb)):
        flip, i = [0] * 6, 0
        for k in range(6):
            if amb[k]:
                flip[k] = (bits >> i) & 1
                i += 1
        signs, offs, spread, _ = solve(tuple(flip))
        r = tool_residual(signs, offs)
        if best is None or r < best[0]:
            best = (r, signs, offs, spread)
    tool_rms, signs, offs, cost = best
    # Named points: p_canonical = x_body + R_body o + lift, linear in (o, lift).
    pts = {}
    lifts = []
    for name, body in bodies.items():
        A, b = [], []
        for q in qs:
            ca.set(q)
            na.set(signs * q + offs)
            x, R = na.body(body)
            A.append(np.hstack([R, np.eye(3)]))
            b.append(ca.point(canon.points[name]) - x)
        sol, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
        pts[name] = {"body": body, "pos": [round(float(v), 6) + 0.0 for v in sol[:3]]}
        lifts.append(-sol[3:])
    return {"joints": [{"canonical": f"joint{k + 1}", "native": j, "sign": int(signs[k]),
                        "offset": round(float(offs[k]), 6) + 0.0} for k, j in enumerate(joints)],
            "points": pts, "lift_m": [round(float(v), 6) for v in np.mean(lifts, axis=0)],
            "offset_spread_rad": cost, "tool_fit_rms_m": tool_rms}


def check(cfg: ModelConfig, canonical: ModelConfig | None = None, n: int = 2000, seed: int = 0,
          lift: np.ndarray | None = None) -> dict:
    """Residuals of ``cfg``'s map against the canonical model (see the module docstring)."""
    canon = canonical or load_config(CANONICAL)
    ca, na = _arm(canon), _arm(cfg)
    jm = JointMap(cfg, na.ranges)
    both = np.stack([np.maximum(ca.ranges[:, 0], jm.canonical_ranges()[:, 0]),
                     np.minimum(ca.ranges[:, 1], jm.canonical_ranges()[:, 1])], axis=1)
    if lift is None:
        ca.set(np.zeros(6))
        na.set(jm.to_native(np.zeros(6), clamp=False))
        (cx, cp), (nx, npos) = ca.axes(), na.axes()
        d = npos[1] - cp[1]
        lift = (d @ cx[0]) * cx[0]       # joint 2 moved along joint 1's axis: the same at every pose
    worst = {"axis_angle_deg": 0.0, "axis_line_mm": np.zeros(6), "axis_line_after_lift_mm": np.zeros(6),
             **{f"{p}_mm": 0.0 for p in cfg.points}, **{f"{p}_after_lift_mm": 0.0 for p in cfg.points}}
    for q in _poses(both, n, seed):
        ca.set(q)
        na.set(jm.to_native(q, clamp=False))
        (cx, cp), (nx, npos) = ca.axes(), na.axes()
        for k in range(6):
            worst["axis_angle_deg"] = max(worst["axis_angle_deg"], float(np.degrees(
                np.arccos(np.clip(cx[k] @ (jm.sign[k] * nx[k]), -1, 1)))))
            for key, shift in (("axis_line_mm", 0.0), ("axis_line_after_lift_mm", lift)):
                dd = npos[k] - shift - cp[k]
                worst[key][k] = max(worst[key][k], 1000 * float(np.linalg.norm(dd - (dd @ cx[k]) * cx[k])))
        for p in cfg.points:
            e = na.point(cfg.points[p]) - ca.point(canon.points[p])
            worst[f"{p}_mm"] = max(worst[f"{p}_mm"], 1000 * float(np.linalg.norm(e)))
            worst[f"{p}_after_lift_mm"] = max(worst[f"{p}_after_lift_mm"], 1000 * float(np.linalg.norm(e - lift)))
    cr, nr = ca.ranges, jm.canonical_ranges()
    return {"model": cfg.name, "poses": n, "lift_mm": [round(1000 * float(v), 3) for v in lift],
            "axis_angle_deg": round(worst["axis_angle_deg"], 6),
            "axis_line_mm": [round(float(v), 3) for v in worst["axis_line_mm"]],
            "axis_line_after_lift_mm": [round(float(v), 3) for v in worst["axis_line_after_lift_mm"]],
            "points_mm": {p: round(worst[f"{p}_mm"], 3) for p in cfg.points},
            "points_after_lift_mm": {p: round(worst[f"{p}_after_lift_mm"], 3) for p in cfg.points},
            "range_differences_rad": {f"joint{k + 1}": {"canonical": [round(float(v), 5) for v in cr[k]],
                                                        "this_model": [round(float(v), 5) for v in nr[k]]}
                                      for k in range(6) if np.abs(cr[k] - nr[k]).max() > 1e-4}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", help="a model config name or path")
    ap.add_argument("--fit", type=Path, help="a one-arm MJCF to fit")
    ap.add_argument("--joints", default=",".join(f"joint{k}" for k in range(1, 7)))
    ap.add_argument("--bodies", help="native bodies for tool,wrist,elbow (comma-separated)")
    ap.add_argument("--finger", help="a native finger joint (fixes the tool roll)")
    args = ap.parse_args(argv)
    if args.check:
        print(json.dumps(check(load_config(args.check)), indent=1))
    if args.fit:
        tool, wrist, elbow = args.bodies.split(",")
        out = fit(args.fit, args.joints.split(","), {"tool": tool, "wrist": wrist, "elbow": elbow}, args.finger)
        print(yaml.safe_dump(out, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
