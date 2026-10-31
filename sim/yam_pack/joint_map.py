"""The joint-mapping layer between the policies and whichever YAM model the simulator runs.

Policies, the oracle and the trace speak one joint convention, the *canonical* one: Robocurve's 14-D
wire format (inspect-robots-yam ``[left j0..j5, left gripper, right j0..j5, right gripper]``), whose
joint values go to i2rt's driver unchanged (``command_joint_pos``), with gripper 1 = open, 0 = closed.
A simulator model may name, zero, orient or limit its joints differently. A *model config*
(``sim/yam_pack/models/<name>.yaml``) says how, and ``JointMap`` converts:

    native = sign * canonical + offset          canonical = sign * (native - offset)

per joint (``sign`` is +1 or -1, so the inverse is the same multiply). Commands outside the native
model's joint range are clamped to it and counted, the way a real driver refuses to drive past a
limit; the count is part of the episode's record, never hidden.

A model config also names the bodies the scene needs (the envelope's named points ``tool``, ``wrist``
and ``elbow`` as physical points on the arm, the body a grasped object is welded to, the finger
joints and their open positions), so swapping models is a config change: the scene, the policies and
the oracle do not change. ``fit.py`` derives a config's numbers from a model file and checks them.

No MuJoCo import here, so the Isaac Lab-Arena driver and the oracle's tests can use it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
CANONICAL = "menagerie_yam"
ARM_DOF = 6
CANONICAL_JOINTS = tuple(f"joint{j}" for j in range(1, ARM_DOF + 1))


@dataclass(frozen=True)
class JointSpec:
    canonical: str
    native: str
    sign: float = 1.0
    offset: float = 0.0


@dataclass
class ModelConfig:
    name: str
    model: Path                      # MJCF of one arm, absolute
    joints: list[JointSpec]
    points: dict[str, dict]          # envelope point -> {site: name} | {body: name, pos: [x, y, z]}
    weld_body: str                   # the body a grasped object is welded to (the gripper mount)
    fingers: list[dict]              # [{joint, open}], held at ``open`` (the fingers never close)
    source: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def is_identity(self) -> bool:
        return all(j.native == j.canonical and j.sign == 1.0 and j.offset == 0.0 for j in self.joints)


def load_config(name_or_path: str | Path | None = None) -> ModelConfig:
    """A model config by name (``models/<name>.yaml``) or path; None means the canonical model."""
    p = Path(name_or_path) if name_or_path is not None and Path(name_or_path).suffix == ".yaml" \
        else MODELS / f"{name_or_path or CANONICAL}.yaml"
    raw = yaml.safe_load(p.read_text())
    joints = [JointSpec(canonical=j["canonical"], native=j["native"], sign=float(j.get("sign", 1)),
                        offset=float(j.get("offset", 0.0))) for j in raw["joints"]]
    if [j.canonical for j in joints] != list(CANONICAL_JOINTS):
        raise ValueError(f"{p.name}: joints must list {CANONICAL_JOINTS} in order")
    if any(j.sign not in (1.0, -1.0) for j in joints):
        raise ValueError(f"{p.name}: every sign must be +1 or -1")
    if set(raw["points"]) != {"tool", "wrist", "elbow"}:
        raise ValueError(f"{p.name}: points must be tool, wrist and elbow")
    model = Path(raw["model"])
    return ModelConfig(name=raw["name"], model=model if model.is_absolute() else (p.parent / model).resolve(),
                       joints=joints, points=raw["points"], weld_body=raw["weld_body"],
                       fingers=raw.get("fingers", []), source=raw.get("source", {}), path=p)


class JointMap:
    """Canonical <-> native joint vectors for one arm (6 joints).

    ``ranges`` are the native model's joint ranges (6 x 2, radians); without them nothing is clamped.
    ``clips`` counts commanded joint values that had to be clamped.
    """

    def __init__(self, config: ModelConfig, ranges: np.ndarray | None = None):
        self.config = config
        self.sign = np.array([j.sign for j in config.joints])
        self.offset = np.array([j.offset for j in config.joints])
        self.native_names = [j.native for j in config.joints]
        self.ranges = None if ranges is None else np.asarray(ranges, float).reshape(ARM_DOF, 2)
        self.clips = 0

    def to_native(self, q: np.ndarray, clamp: bool = True) -> np.ndarray:
        n = self.sign * np.asarray(q, float) + self.offset
        if clamp and self.ranges is not None:
            c = np.clip(n, self.ranges[:, 0], self.ranges[:, 1])
            self.clips += int(np.count_nonzero(c != n))
            n = c
        return n

    def to_canonical(self, q_native: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(q_native, float) - self.offset)

    def canonical_ranges(self) -> np.ndarray | None:
        """The native ranges expressed in the canonical convention (lo <= hi)."""
        if self.ranges is None:
            return None
        r = np.stack([self.to_canonical(self.ranges[:, 0]), self.to_canonical(self.ranges[:, 1])], axis=1)
        return np.sort(r, axis=1)
