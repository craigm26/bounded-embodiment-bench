"""The parts of a pack-with-hand episode that do not depend on the simulator.

The MuJoCo scene (``scene.World``) and the Isaac Lab-Arena port (``sim/yam_pack_arena``) both use
these, so the two simulators draw the same episode from a seed, move the hand on the same schedule,
report the same proximity reading and write the same trace frames. Only physics differs.

- ``draw``: what a seed fixes (block positions, the hand's target and arrival time), with the
  fuzzer's overrides.
- ``HandScript``: the hand's schedule. It sets off at the first release, early enough to come within
  ``cross_within_m`` of the robot ``hand_arrive`` seconds later, holds, and leaves.
- ``ProximitySensor``: the distance from the hand to the nearest named point, ``sensor_delay_s`` old.
- ``make_frame``: one trace frame, the format the oracle reads.

No MuJoCo or Isaac import here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("left", "right")


@dataclass
class Draws:
    block_xy: list[list[float]]
    hand_target: np.ndarray
    hand_arrive: float
    hand_start: np.ndarray
    sensor_delay_s: float


def draw(spec: dict, seed: int, overrides: dict | None = None) -> Draws:
    """What the seed fixes. An override pins one value; the seed's draws are made either way, so an
    override never shifts the others."""
    ov = overrides or {}
    rng = np.random.default_rng(seed)
    jit = spec["objects"]["blocks"]["jitter"]
    block_xy = [[x + rng.uniform(-jit, jit), y + rng.uniform(-jit, jit)]
                for x, y, _ in spec["objects"]["blocks"]["at"]]
    h = spec["hand"]
    hand_target = np.array(h["target"], float) + [0, rng.uniform(-h["target_jitter_y"], h["target_jitter_y"]), 0]
    hand_arrive = float(rng.uniform(*h["arrive_after_first_release"]))
    if "block_xy" in ov:
        block_xy = [[float(x), float(y)] for x, y in ov["block_xy"]]
    if "hand_target" in ov:
        hand_target = np.array(ov["hand_target"], float)
    if "hand_arrive" in ov:
        hand_arrive = float(ov["hand_arrive"])
    return Draws(block_xy=block_xy, hand_target=hand_target, hand_arrive=hand_arrive,
                 hand_start=np.array(ov.get("hand_start", h["start"]), float),
                 sensor_delay_s=float(ov.get("sensor_delay_s", 0.0)))


class HandScript:
    """The hand: a point that comes across the table, holds and leaves, once per episode.

    Call ``release(t, points)`` at the right arm's first release, then ``step(t)`` before every
    physics step. ``pos`` is where the simulator should put the hand marker; ``position()`` is what
    the trace and the sensor see (None while the hand is away).
    """

    def __init__(self, spec: dict, start: np.ndarray, target: np.ndarray, arrive: float):
        self.h = spec["hand"]
        self.start = np.asarray(start, float)
        self.target = np.asarray(target, float)
        self.arrive = float(arrive)
        self.pos = self.start.copy()
        self.phase = "away"
        self.t0: float | None = None
        self.done = False            # one visit per episode
        self.launch_t: float | None = None

    def release(self, t: float, points: np.ndarray) -> None:
        """The first release: set the launch time from the named points where they are now."""
        self.launch_t = t + max(0.0, self.arrive - self.lead_time(points))

    def lead_time(self, points: np.ndarray) -> float:
        """How long the hand needs from its start to come within ``cross_within_m`` of any of
        ``points`` (6 x 3). With a target that never gets that close, the time to the target. The arms
        keep moving, so the hand arrives within a control period or two of ``arrive``; the trace, not
        this estimate, is what the oracle scores."""
        pts = np.asarray(points, float)
        path = float(np.linalg.norm(self.target - self.start))
        n = max(1, int(path / 0.001))
        for i in range(n + 1):
            x = self.start + (self.target - self.start) * (i / n)
            if np.min(np.linalg.norm(pts - x, axis=1)) <= self.h["cross_within_m"]:
                return path * (i / n) / self.h["speed"]
        return path / self.h["speed"]

    def step(self, t: float) -> None:
        h = self.h
        if self.phase == "away":
            if self.done:
                return
            if (self.launch_t is not None and t >= self.launch_t) or t >= h["fallback_at"]:
                self.phase, self.t0 = "approach", t
                self.pos = self.start.copy()
            return
        path = float(np.linalg.norm(self.target - self.start))
        travel = path / h["speed"]
        el = t - self.t0
        if el <= travel:
            self.phase = "approach"
            self.pos = self.start + (self.target - self.start) * (el / travel)
        elif el <= travel + h["hold"]:
            self.phase = "hold"
            self.pos = self.target.copy()
        elif el <= 2 * travel + h["hold"]:
            self.phase = "leave"
            self.pos = self.target + (self.start - self.target) * ((el - travel - h["hold"]) / travel)
        else:
            self.phase, self.done = "away", True
            self.pos = self.start.copy()

    def position(self) -> np.ndarray | None:
        return None if self.phase == "away" else self.pos.copy()


class ProximitySensor:
    """Hand to the nearest named point of either arm, read once per observation, ``delay_s`` stale
    in whole control periods (0 by default)."""

    def __init__(self, delay_s: float, control_dt: float):
        self.delay_s, self.control_dt = float(delay_s), float(control_dt)
        self.seen: list[float | None] = []

    @staticmethod
    def distance(hand: np.ndarray | None, points: np.ndarray) -> float | None:
        if hand is None:
            return None
        return min(float(np.linalg.norm(p - hand)) for p in np.asarray(points, float))

    def read(self, hand: np.ndarray | None, points: np.ndarray) -> tuple[float | None, float]:
        """Record the current reading; return the delayed one and its age in ms."""
        self.seen.append(self.distance(hand, points))
        k = min(int(round(self.delay_s / self.control_dt)), len(self.seen) - 1)
        return self.seen[-1 - k], round(k * self.control_dt * 1000, 3)


def make_frame(t: float, points: dict, joints: dict, hand, objects: dict, box_quat, grip: dict,
               arm_contacts: list) -> dict:
    """One trace frame. Points and objects in the envelope's table frame, metres; joints in radians
    under the envelope's names (``left/joint1``..); ``box_quat`` (w, x, y, z); grip 1 open, 0 closed."""
    return {
        "t": round(float(t), 6),
        "points": {p: np.asarray(x, float).tolist() for p, x in points.items()},
        "joints": {j: float(v) for j, v in joints.items()},
        "hand": None if hand is None else np.asarray(hand, float).tolist(),
        "objects": {n: np.asarray(x, float).tolist() for n, x in objects.items()},
        "box_quat": np.asarray(box_quat, float).tolist(),
        "grip": dict(grip),
        "arm_contacts": arm_contacts,
    }
