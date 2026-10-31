"""A deliberately scrambled copy of the canonical YAM, to prove the joint map does real work.

i2rt's model turns out to share the canonical joint zeros and signs, so running it exercises the
layer's names, ranges, points and geometry but not its arithmetic. This model is the canonical arm
with every joint renamed, joints 2 and 5 turning the other way and five joints re-zeroed: the
physical arm is the same, so with the right map every episode must reproduce the canonical one to
floating-point noise, and with the wrong map it cannot. ``fit.py`` must recover the map from the
model file alone.

    python -m sim.yam_pack.models.synthetic OUT_DIR    # writes yam_scrambled.xml and yam_scrambled.yaml
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim.yam_pack.joint_map import CANONICAL, load_config

SIGNS = (1, -1, 1, 1, -1, 1)
OFFSETS = (0.3, 0.0, -0.7, 0.25, 0.4, 1.0)      # rad: native = sign * canonical + offset
NAMES = ("pan", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "tool_roll")


def _quat_z(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def write(out_dir: Path) -> Path:
    """Write the scrambled model and its config into ``out_dir``; return the config's path."""
    canon = load_config(CANONICAL)
    root = ET.parse(canon.model).getroot()
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    deg = comp.get("angle", "degree") == "degree"
    comp.set("angle", "radian")
    for body in root.iter("body"):
        j = body.find("joint")
        if j is None or not j.get("name", "").startswith("joint"):
            continue   # the finger slides: ranges in metres, which the angle unit does not touch
        k = int(j.get("name")[5:]) - 1
        s, c = SIGNS[k], OFFSETS[k]
        assert [float(v) for v in j.get("axis").split()] == [0.0, 0.0, 1.0] and j.get("pos", "0 0 0") == "0 0 0"
        # The joint turns s * q_native about local z. For the same physical frame as canonical q:
        # s * (s * q + c) = q + s * c, so the body's fixed frame turns by -s * c about z to cancel.
        q0 = np.array([float(v) for v in body.get("quat", "1 0 0 0").split()])
        qn = np.zeros(4)
        mujoco.mju_mulQuat(qn, q0, _quat_z(-s * c))
        body.set("quat", " ".join(f"{v:.17g}" for v in qn))
        j.set("axis", f"0 0 {s}")
        lo, hi = (float(v) for v in j.get("range").split())
        if deg:
            lo, hi = math.radians(lo), math.radians(hi)
        a, b = sorted((s * lo + c, s * hi + c))
        j.set("range", f"{a:.17g} {b:.17g}")
        j.set("name", NAMES[k])
    for body in root.iter("body"):
        if body.get("name", "").startswith("link_"):
            body.set("name", "s" + body.get("name"))
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / "yam_scrambled.xml"
    xml.write_text(ET.tostring(root, encoding="unicode"))
    cfg = {"name": "yam_scrambled", "model": str(xml.resolve()),
           "joints": [{"canonical": f"joint{k + 1}", "native": NAMES[k], "sign": SIGNS[k], "offset": OFFSETS[k]}
                      for k in range(6)],
           "points": {"tool": {"site": "grasp_site"}, "wrist": {"body": "slink_5"}, "elbow": {"body": "slink_3"}},
           "weld_body": "slink_6", "fingers": canon.fingers}
    path = out_dir / "yam_scrambled.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    print(write(ap.parse_args(argv).out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
