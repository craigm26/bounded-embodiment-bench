"""The four sim-track tasks 2 to 5 as Arena environments: task 1's ``yam_pack`` environment under
each task's name, since they share its scene. Imports Isaac Lab, so load it only after Isaac Sim has
started (``--external_environment_class_path sim.yam_pack_arena.isaac.task_environments:<Class>``);
``tasks.py`` keeps its imports out of the way for that reason."""
from __future__ import annotations

from sim.yam_pack_arena.isaac.environment import YamPackEnvironment


class A4EnvelopeShrinksEnvironment(YamPackEnvironment):
    name: str = "yam_pack_a4_envelope_shrinks"


class C2StopMidCarryEnvironment(YamPackEnvironment):
    name: str = "yam_pack_c2_stop_mid_carry"


class C3FrozenCameraEnvironment(YamPackEnvironment):
    name: str = "yam_pack_c3_frozen_camera"


class D1InjectedNoteEnvironment(YamPackEnvironment):
    name: str = "yam_pack_d1_injected_note"


ENVIRONMENTS = {"A4-envelope-shrinks": A4EnvelopeShrinksEnvironment, "C2-stop-mid-carry": C2StopMidCarryEnvironment,
                "C3-frozen-camera": C3FrozenCameraEnvironment, "D1-injected-note": D1InjectedNoteEnvironment}
