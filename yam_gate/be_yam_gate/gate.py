# SPDX-FileCopyrightText: 2026 Craig Merry and contributors
# SPDX-License-Identifier: MIT
"""A Bounded Embodiment gate for a bimanual YAM, placed between a model (Dream-1) and the arms.

It is a port of the method's reference gate idea (examples/reference-gates/so101_reference.py in
RobotRegistryFoundation/bounded-embodiment) to the YAM's 14-D joint-target action, using the method's own envelope
loader and clearance: ``bounded_embodiment.envelope.load`` and ``Envelope.clearance(point_id, xyz)``.

Every command row, for each arm:
  1. Validate: 14 finite numbers. Invalid: hold (re-send the reference).
  2. Clamp each joint to the URDF's limits, and cap each joint's step at the envelope's
     ``motion.max_joint_speed_radps`` (if declared) for one control period.
  3. Predictive workspace check: sample the joint-space line from the reference (the joint goals last sent) toward the
     clamped goal at ``SAMPLES`` fractions; at each, forward kinematics gives the arm's declared points (the tool point,
     and any link points the envelope names) in the envelope's frame; ``Envelope.clearance`` must stay at or above
     ``margin_m`` for every point. The command is cut at the last fraction that passes; if the first fails, the arm holds.
     A point that is already outside (clearance below the margin at the reference) may only move to increase its
     clearance, so the gate can recover an arm that started outside.
  4. Speed: the tool point's displacement per control period may not exceed the point's ``max_speed_mps`` times the
     period; a longer step is scaled down along the same joint-space line.
  5. Grippers pass through unchanged. The gate bounds where the hand can go, not what it lets go of: a keep-out whose
     height band reaches the carry height is what stops a model releasing an object over it.

The gate keeps the reference it last sent and plans from it (the reference gate's lesson: planning from the measured
pose re-anchors the goal to a sagging servo every cycle). It reports every intervention, so a run can say how often the
model asked for something the envelope forbids.

AN EXAMPLE, NOT A PRODUCT: it exists to make the declared limits concrete on Robocurve's simulated YAM tasks. It runs
only in simulation; it has not been measured against a real arm, and its results are not EV-03 or EV-10 results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from bounded_embodiment.envelope import load as load_envelope

SAMPLES = 8
ARMS = ("left", "right")
JOINTS = 6
DIM = 14  # per arm [j0..j5, gripper], left then right


@dataclass
class Decision:
    arm: str
    kind: str  # "pass", "cut", "hold", "invalid", "slowed"
    fraction: float
    min_clearance_m: float
    point: str | None = None


@dataclass
class GateStats:
    rows: int = 0
    passed: int = 0
    cut: int = 0
    held: int = 0
    slowed: int = 0
    invalid: int = 0
    worst_requested_clearance_m: float = math.inf
    by_zone: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (v if not isinstance(v, float) or math.isfinite(v) else None) for k, v in self.__dict__.items()}


class YamEnvelopeGate:
    """``gate.filter(row)`` returns the row to send. ``fk(arm, q6)`` returns {point_id: xyz} in the envelope frame."""

    def __init__(self, envelope_path: str | Path, fk, lower, upper, control_hz: float, margin_m: float = 0.005):
        self.envelope = load_envelope(str(envelope_path))
        self.fk = fk
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.dt = 1.0 / float(control_hz)
        self.margin = float(margin_m)
        motion = self.envelope.motion
        self.joint_step = (motion.get("max_joint_speed_radps") or math.inf) * self.dt
        self.points = {arm: [p for p in self.envelope.points if p.startswith(arm + "_")] for arm in ARMS}
        for arm in ARMS:
            if f"{arm}_tool" not in self.envelope.points:
                raise ValueError(f"the envelope must declare a point '{arm}_tool'")
        self.ref: np.ndarray | None = None
        self.stats = GateStats()
        self.log: list[Decision] = []

    # -- geometry ------------------------------------------------------------------------------------------------

    def _clearances(self, arm: str, q6: np.ndarray) -> dict[str, float]:
        xyz = self.fk(arm, q6)
        return {pid: self.envelope.clearance(pid, xyz[pid]) for pid in self.points[arm] if pid in xyz}

    def _ok(self, arm: str, q6: np.ndarray, start: dict[str, float]) -> tuple[bool, float, str | None]:
        cl = self._clearances(arm, q6)
        worst, who = math.inf, None
        for pid, c in cl.items():
            floor = self.margin if start[pid] >= self.margin else start[pid]  # outside already: may not get worse
            if c < floor - 1e-9:
                return False, c, pid
            if c < worst:
                worst, who = c, pid
        return True, worst, who

    # -- the gate ------------------------------------------------------------------------------------------------

    def reset(self, measured_row) -> None:
        self.ref = np.asarray(measured_row, float).copy()
        self.stats = GateStats()
        self.log = []

    def filter(self, row) -> np.ndarray:
        row = np.asarray(row, float)
        if self.ref is None:
            self.reset(row)
            return row.copy()
        self.stats.rows += 1
        if row.shape != (DIM,) or not np.all(np.isfinite(row)):
            self.stats.invalid += 1
            self.log.append(Decision("both", "invalid", 0.0, math.nan))
            return self.ref.copy()
        out = self.ref.copy()
        for a, arm in enumerate(ARMS):
            i = a * 7
            q0 = self.ref[i:i + JOINTS]
            goal = np.clip(row[i:i + JOINTS], self.lower, self.upper)
            step = np.clip(goal - q0, -self.joint_step, self.joint_step)
            goal = q0 + step
            start = self._clearances(arm, q0)
            # requested-pose clearance, for the report
            req = self._clearances(arm, np.clip(row[i:i + JOINTS], self.lower, self.upper))
            if req:
                self.stats.worst_requested_clearance_m = min(self.stats.worst_requested_clearance_m, min(req.values()))
            best, worst_c, who, kind = 0.0, math.inf, None, "pass"
            for k in range(1, SAMPLES + 1):
                f = k / SAMPLES
                ok, c, pid = self._ok(arm, q0 + f * (goal - q0), start)
                if not ok:
                    kind, worst_c, who = ("hold" if best == 0.0 else "cut"), c, pid
                    break
                best, worst_c, who = f, min(worst_c, c), pid
            q = q0 + best * (goal - q0)
            # speed: tool displacement per period
            tool = f"{arm}_tool"
            vmax = self.envelope.speed_limit_mps(tool)
            p0 = np.asarray(self.fk(arm, q0)[tool])
            limit = vmax * self.dt
            for _ in range(6):  # joint-space scaling is not exactly linear in tool distance: iterate until within
                d = float(np.linalg.norm(np.asarray(self.fk(arm, q)[tool]) - p0))
                if d <= limit or limit <= 0:
                    break
                q = q0 + (q - q0) * (0.999 * limit / d)
                if kind == "pass":
                    kind = "slowed"
            out[i:i + JOINTS] = q
            out[i + JOINTS] = row[i + JOINTS]  # gripper passes through
            if kind != "pass":
                self.log.append(Decision(arm, kind, best, worst_c, who))
                if who:
                    self.stats.by_zone[who] = self.stats.by_zone.get(who, 0) + 1
            counter = {"pass": "passed", "cut": "cut", "hold": "held", "slowed": "slowed"}[kind]
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)
        self.ref = out.copy()
        return out
