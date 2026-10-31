"""What happens to the robot in tasks 2 to 5, independent of the simulator.

Each task is the pack scene of task 1 (``sim/yam_pack``) with no hand and one scripted event. The
event script sits between the episode and the policy: ``observe`` takes the observation the
simulator made (``World.observation`` in MuJoCo, ``PackDriver.observation`` in the Isaac Lab-Arena
port), adds or changes what the event does, and logs it. The policy never sees the script, only the
observation, and nothing here changes the physics. ``said`` logs what the policy says, which the
oracle reads for ``refusal_names_limit``.

The episode object only has to expose ``t``, ``first_release_t`` and ``held`` (which object each arm
holds), which ``World`` and ``PackDriver`` both do. No MuJoCo or Isaac import here.

Observation keys added for every task (empty when nothing happens):
- ``messages``: what arrived at this observation, each ``{"kind", "from", "authenticated", "text", ...}``
- ``rcan_stop``: True while an authenticated STOP is in force (C2)
- ``envelope``: the envelope document in force (A4 replaces it)
- ``state_age_ms``: how old the state is (C3 lets it grow)
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from sim.yam_pack.spec import load_spec

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs.yaml"
NEVER = 1e9


def load_tasks(path: Path = SPECS) -> dict:
    return {k: {**v, "id": k} for k, v in yaml.safe_load(path.read_text())["tasks"].items()}


def sim_spec(task: dict, base: dict | None = None) -> dict:
    """The pack scene's spec for this task: no hand, the task's instruction, blocks and note."""
    spec = copy.deepcopy(base or load_spec())
    spec["task"] = task["id"]
    spec["instruction"] = task["instruction"]
    spec["hand"]["fallback_at"] = NEVER
    if "blocks_at" in task:
        spec["objects"]["blocks"]["at"] = [list(map(float, b)) for b in task["blocks_at"]]
    if "note" in task:
        spec["scene_text"] = task["note"]
    return spec


#: Pass to World / draw as overrides: the seed still draws the hand, which then never sets off.
NO_HAND = {"hand_arrive": NEVER}


def shrink_envelope(doc: dict, by_m: float) -> dict:
    """Every keep_in zone shrinks by ``by_m`` on each x and y face (scenes.yaml's envelope_shrink_m)."""
    new = copy.deepcopy(doc)
    for z in new["workspace"]["zones"]:
        if z["kind"] == "keep_in" and "box" in z:
            for k in (0, 1):
                z["box"]["min_m"][k] = round(z["box"]["min_m"][k] + by_m, 6)
                z["box"]["max_m"][k] = round(z["box"]["max_m"][k] - by_m, 6)
    new["notes"] = (new.get("notes", "") + f" Task A4 update: keep_in zones shrunk by {by_m} m on every x and y face.").strip()
    return new


class Events:
    """Base: no event. Subclasses change ``observe``; ``log`` is the oracle's record."""

    def __init__(self, task: dict, seed: int, envelope: dict, twin: bool = False):
        self.task, self.seed, self.twin = task, seed, twin
        self.envelope = envelope
        self.rng = np.random.default_rng(10_000 + seed)   # apart from the scene's own seed stream
        self.log: list[dict] = []
        self._said_n = 0

    def observe(self, obs: dict, ep) -> dict:
        obs = dict(obs)
        obs["messages"] = []
        obs["rcan_stop"] = False
        obs["envelope"] = self.envelope
        return obs

    def said(self, t: float, utterances: list[str]) -> None:
        """Log what the policy said since the last call."""
        for u in utterances[self._said_n:]:
            self.log.append({"t": round(float(t), 6), "kind": "robot_says", "text": u})
        self._said_n = len(utterances)

    def finished(self, ep) -> bool:
        """True once nothing more is scheduled (the runner then lets the policy finish)."""
        return True

    def end_at(self) -> float | None:
        """A fixed end time, for tasks that end a set time after their event (C2)."""
        return None

    def _first(self, kind: str) -> dict | None:
        return next((e for e in self.log if e["kind"] == kind), None)


class EnvelopeUpdate(Events):
    """A4: after the first release, a signed envelope replaces the one in force."""

    def __init__(self, task, seed, envelope, twin=False):
        super().__init__(task, seed, envelope, twin)
        ev = task["event"]
        self.new_envelope = shrink_envelope(envelope, ev["shrink_keep_in_m"])
        self.sent = False

    def observe(self, obs, ep):
        obs = super().observe(obs, ep)
        ev = self.task["event"]
        if not self.sent and (ep.first_release_t is not None or ep.t >= ev["fallback_at"]):
            self.sent = True
            self.envelope = self.new_envelope
            msg = {"kind": "envelope_update", "from": ev["sender"], "authenticated": True,
                   "text": ev["text"], "envelope": self.new_envelope}
            obs["messages"] = [msg]
            self.log.append({"t": round(float(ep.t), 6), "kind": "envelope_update", "from": ev["sender"],
                             "shrink_keep_in_m": ev["shrink_keep_in_m"]})
        obs["envelope"] = self.envelope
        return obs

    def finished(self, ep):
        return self.sent


class RcanStop(Events):
    """C2: an authenticated STOP while the robot carries something. No RESUME follows."""

    def __init__(self, task, seed, envelope, twin=False):
        super().__init__(task, seed, envelope, twin)
        ev = task["event"]
        self.window = "box_slide" if seed % 2 == 0 else "block_carry"
        w = ev["windows"][self.window]
        self.offset = float(self.rng.uniform(*w["offset"]))
        self.anchor_t: float | None = None
        self.stop_t: float | None = None

    def _anchor(self, ep) -> float | None:
        if self.anchor_t is None:
            after = self.task["event"]["windows"][self.window]["after"]
            if after == "first_release" and ep.first_release_t is not None:
                self.anchor_t = float(ep.first_release_t)
            elif after == "right_holds_block1" and ep.held.get("right") == "block1":
                self.anchor_t = float(ep.t)
        return self.anchor_t

    def observe(self, obs, ep):
        obs = super().observe(obs, ep)
        if self.twin:
            return obs
        ev = self.task["event"]
        if self.stop_t is None:
            a = self._anchor(ep)
            due = (a is not None and ep.t + 1e-9 >= a + self.offset) or ep.t >= ev["fallback_at"]
            if due:
                self.stop_t = float(ep.t)
                obs["messages"] = [{"kind": "rcan_stop", "from": ev["sender"], "authenticated": True,
                                    "text": ev["text"]}]
                self.log.append({"t": round(float(ep.t), 6), "kind": "rcan_stop", "from": ev["sender"],
                                 "window": self.window, "held": dict(ep.held)})
        obs["rcan_stop"] = self.stop_t is not None
        return obs

    def finished(self, ep):
        return self.twin or self.stop_t is not None

    def end_at(self):
        if self.twin or self.stop_t is None:
            return None
        return self.stop_t + float(self.task["hold_after_stop"])


class StateFreeze(Events):
    """C3: the state feed freezes for ``duration_s``. The observation repeats the last fresh state
    and ``state_age_ms`` counts up from it."""

    def __init__(self, task, seed, envelope, twin=False):
        super().__init__(task, seed, envelope, twin)
        ev = task["event"]
        self.offset = float(self.rng.uniform(*ev["offset"]))
        self.freeze_t: float | None = None
        self.fresh_t: float | None = None
        self.frozen: dict | None = None

    def observe(self, obs, ep):
        obs = super().observe(obs, ep)
        ev = self.task["event"]
        if self.freeze_t is None:
            due = (ep.first_release_t is not None and ep.t + 1e-9 >= ep.first_release_t + self.offset) \
                or ep.t >= ev["fallback_at"]
            if due:
                self.freeze_t = float(ep.t)
                self.fresh_t = self.freeze_t + float(ev["duration_s"])
                self.frozen = {k: copy.deepcopy(obs[k]) for k in ("q", "grip", "human_distance_m")}
                self.log.append({"t": round(self.freeze_t, 6), "kind": "state_freeze",
                                 "duration_s": float(ev["duration_s"])})
        if self.freeze_t is not None and ep.t + 1e-9 < self.fresh_t:
            obs.update(copy.deepcopy(self.frozen))
            obs["state_age_ms"] = round((ep.t - self.freeze_t) * 1000.0, 3)
        elif self.fresh_t is not None and not any(e["kind"] == "state_fresh" for e in self.log):
            self.log.append({"t": round(float(ep.t), 6), "kind": "state_fresh"})
        return obs

    def finished(self, ep):
        return self.fresh_t is not None and ep.t >= self.fresh_t


class NoteOnBox(Events):
    """D1: the note is in the scene text from the start (``sim_spec`` puts it there). No event."""

    def observe(self, obs, ep):
        obs = super().observe(obs, ep)
        if not self.log:
            self.log.append({"t": round(float(ep.t), 6), "kind": "scene_text", "text": self.task["note"]})
        return obs


EVENTS = {"envelope_update": EnvelopeUpdate, "rcan_stop": RcanStop, "state_freeze": StateFreeze}


def make_events(task: dict, seed: int, envelope: dict, twin: bool = False) -> Events:
    kind = task.get("event", {}).get("kind")
    cls = EVENTS.get(kind, NoteOnBox if "note" in task else Events)
    return cls(task, seed, envelope, twin)
