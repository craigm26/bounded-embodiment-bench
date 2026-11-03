# SPDX-FileCopyrightText: 2026 Craig Merry and contributors
# SPDX-License-Identifier: MIT
"""Inspect Robots policy ``dream_be_gated``: Dream-1, with every command row passed through a Bounded Embodiment gate.

    inspect-robots run --embodiment yam_arena --policy dream_be_gated -P envelope=/path/task.envelope.json ...

Same model, same scene, same instruction as ``--policy dream``; the only difference is the gate between Dream-1 and the
arms. The gate's interventions go into the trial transcript (``gate``), so a run reports how often Dream-1 asked for
something the declared envelope forbids.
"""

from __future__ import annotations

from typing import Any

from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_dream.policy import DreamPolicy, _measured

from be_yam_gate.gate import YamEnvelopeGate
from be_yam_gate.yam import JOINT_LOWER, JOINT_UPPER, fk


class GatedDreamPolicy(DreamPolicy):
    def __init__(self, *, envelope: str, margin_m: float = 0.005, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.envelope_path = envelope
        self.margin_m = float(margin_m)
        self.gate: YamEnvelopeGate | None = None
        self.info.name = "dream_be_gated"

    def bind(self, embodiment_info) -> None:
        super().bind(embodiment_info)
        self.gate = YamEnvelopeGate(self.envelope_path, fk, JOINT_LOWER, JOINT_UPPER, self._hz, self.margin_m)

    def reset(self, scene) -> None:
        super().reset(scene)
        if self.gate is not None:
            self.gate.ref = None

    def act(self, observation: Observation) -> ActionChunk:
        if self.gate is not None and self.gate.ref is None:
            self.gate.reset(_measured(observation))
        chunk = super().act(observation)
        row = chunk.actions[0].data
        gated = self.gate.filter(row) if self.gate is not None else row
        if self._trial.last is not None:
            self._trial.last = gated.copy()  # Dream is told what was actually commanded
        return ActionChunk(actions=[Action(data=gated)], control_hz=chunk.control_hz,
                           inference_latency_s=chunk.inference_latency_s)

    def transcript(self) -> dict[str, Any]:
        out = super().transcript()
        if self.gate is not None:
            out["gate"] = {
                "envelope": self.envelope_path,
                "envelope_hash": self.gate.envelope.hash,
                "margin_m": self.margin_m,
                "stats": self.gate.stats.as_dict(),
                "interventions": [d.__dict__ for d in self.gate.log[:2000]],
            }
        return out


def dream_be_gated_policy(**kwargs: Any) -> GatedDreamPolicy:
    """Policy ``dream_be_gated``; ``-P envelope=<path>`` (required), ``-P margin_m=0.005``."""
    return GatedDreamPolicy(**kwargs)
