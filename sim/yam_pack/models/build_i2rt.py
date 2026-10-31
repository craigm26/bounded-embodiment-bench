"""Build the scene's copy of i2rt's own YAM model from an i2rt checkout.

    python -m sim.yam_pack.models.build_i2rt /path/to/i2rt   # writes sim/yam_pack/assets/i2rt_yam_v1/

i2rt's release (github.com/i2rt-robotics/i2rt, v1.4.0 = fe78bae) describes the YAM arm in
``i2rt/robot_models/arm/yam/v1/yam.xml`` and the gripper Robocurve's adapter defaults to
(``LINEAR_4310``) in ``i2rt/robot_models/gripper/linear_4310/linear_4310.xml``, mounted with the
transform in ``i2rt/robots/config/linear_4310.yml``. This script reads those three files as data (it
never imports or runs i2rt's code) and does what i2rt's ``combine_arm_and_gripper_xml`` does: the
gripper's children move into the arm's ``gripper`` mount body under a frame carrying the gripper
body's pose, and the gripper's inertial replaces the mount's placeholder.

Then it changes four things, so the model runs in the same scene as the canonical one:

1. Meshes are dropped (9.6 MB of STL). i2rt's model has no other geometry, so
2. collision capsules are added: the canonical model's (Robocurve's reduction of Menagerie), placed on
   the matching i2rt body so each capsule rides the same physical link. Bodies above joint 1 sit
   ``lift`` higher in i2rt's model (its shoulder is 9.5 mm higher), and their capsules move up with
   them. The finger capsules go on the gripper body at the open position (the scene never closes the
   fingers; a grasp is a weld).
3. Gravity compensation on every moving body (``gravcomp="1"``), which i2rt's driver does on the real
   arm (``gravity_comp_factor`` in ``i2rt/robots/config/yam_v1.yml``) and the canonical model has.
4. Motor armature and joint friction from the canonical model's motor classes (DM4340 on joints 1-3,
   DM4310 on 4-6, the motors i2rt's ``yam_v1.yml`` lists). i2rt's MJCF has none.

Kinematics, joint ranges, inertials, ``actuatorfrcrange`` and sites are i2rt's, unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
OUT_DIR = ASSETS / "i2rt_yam_v1"
OUT = OUT_DIR / "yam_linear_4310.xml"
CANONICAL_XML = ASSETS / "yam_collision.xml"
ARM = "i2rt/robot_models/arm/yam/v1/yam.xml"
GRIPPER = "i2rt/robot_models/gripper/linear_4310/linear_4310.xml"
MOUNT = "i2rt/robots/config/linear_4310.yml"
# canonical body -> i2rt body carrying its capsules. Finger bodies collapse onto the gripper.
BODY_MAP = {"arm": "base", "link_2": "link2", "link_3": "link3", "link_4": "link4", "link_5": "link5",
            "link_6": "gripper", "link_left_finger": "gripper", "lf_rot": "gripper", "lf_down": "gripper",
            "link_right_finger": "gripper", "rf_rot": "gripper", "rf_down": "gripper"}
MOTORS = {"joint1": "dm4340", "joint2": "dm4340", "joint3": "dm4340",
          "joint4": "dm4310", "joint5": "dm4310", "joint6": "dm4310"}
ARMATURE = {"dm4340": 0.032, "dm4310": 0.0018, "finger": 0.1}   # canonical model's classes
FRICTIONLOSS = 0.1


def _f(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def _s(v) -> str:
    return " ".join(f"{float(x):.10g}" for x in np.asarray(v).ravel())


def _compose(op, oq, ip, iq):
    r = np.zeros(3)
    mujoco.mju_rotVecQuat(r, np.asarray(ip, float), np.asarray(oq, float))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(oq, float), np.asarray(iq, float))
    return np.asarray(op, float) + r, q


def merge(i2rt: Path) -> ET.Element:
    """i2rt's arm + LINEAR_4310 gripper as one MJCF, meshes dropped (steps 0 and 1 above)."""
    arm = ET.parse(i2rt / ARM).getroot()
    grip = ET.parse(i2rt / GRIPPER).getroot()
    mount = yaml.safe_load((i2rt / MOUNT).read_text())["last_joint_mount"]["yam"]["v1"]
    for root in (arm, grip):
        for asset in root.findall("asset"):
            root.remove(asset)
        for parent in root.iter():
            for g in [c for c in parent if c.tag == "geom" and c.get("type") == "mesh"]:
                parent.remove(g)
        comp = root.find("compiler")
        if comp is not None and "meshdir" in comp.attrib:
            del comp.attrib["meshdir"]
    body = arm.find("worldbody")
    while [c for c in body if c.tag == "body"]:
        body = [c for c in body if c.tag == "body"][0]
    assert body.get("name") == "gripper", body.get("name")
    body.set("pos", mount["pos"])
    body.set("quat", mount["quat"])
    body.find("joint").set("axis", mount["axis"])
    gb = grip.find(".//body[@name='gripper']")
    gp, gq = gb.get("pos", "0 0 0"), gb.get("quat", "1 0 0 0")
    frame = ET.Element("frame", {"pos": gp, "quat": gq})
    frame.extend(deepcopy(c) for c in gb if c.tag != "inertial")
    body.append(frame)
    gi = gb.find("inertial")
    body.remove(body.find("inertial"))
    merged = deepcopy(gi)
    p, q = _compose(_f(gp), _f(gq), _f(gi.get("pos", "0 0 0")), _f(gi.get("quat", "1 0 0 0")))
    merged.set("pos", _s(p))
    merged.set("quat", _s(q))
    body.insert(0, merged)
    for sec in ("equality", "contact"):
        if grip.find(sec) is not None:
            arm.append(deepcopy(grip.find(sec)))
    return arm


def _q0(m: mujoco.MjModel, open_fingers: dict[str, float] | None = None) -> mujoco.MjData:
    d = mujoco.MjData(m)
    for j, v in (open_fingers or {}).items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    mujoco.mj_kinematics(m, d)
    return d


def shoulder_lift(mi: mujoco.MjModel, mc: mujoco.MjModel) -> np.ndarray:
    """How far i2rt's joint-2 axis sits from the canonical one at q = 0, as a vector across the axis.
    It is the same at every pose because it lies along the joint 1 axis (z)."""
    di, dc = _q0(mi), _q0(mc)
    a = mi.joint("joint2").id
    b = mc.joint("joint2").id
    d = di.xanchor[a] - dc.xanchor[b]
    ax = dc.xaxis[b]
    return d - (d @ ax) * ax


def add_collision(root: ET.Element, lift: np.ndarray) -> int:
    """Step 2: the canonical model's capsules on the matching i2rt bodies."""
    mi = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    mc = mujoco.MjModel.from_xml_path(str(CANONICAL_XML))
    di = _q0(mi)
    dc = _q0(mc, {"left_finger": mc.jnt_range[mc.joint("left_finger").id][1],
                  "right_finger": mc.jnt_range[mc.joint("right_finger").id][0]})
    bodies = {b.get("name"): b for b in root.iter("body")}
    n = 0
    for g in range(mc.ngeom):
        src = mc.body(mc.geom_bodyid[g]).name
        dst = BODY_MAP[src]
        gp = dc.geom_xpos[g] + (lift if src != "arm" else 0.0)
        gq = np.zeros(4)
        mujoco.mju_mat2Quat(gq, dc.geom_xmat[g])
        b = mi.body(dst).id
        biq = np.zeros(4)
        mujoco.mju_negQuat(biq, di.xquat[b])
        lp = np.zeros(3)
        mujoco.mju_rotVecQuat(lp, gp - di.xpos[b], biq)
        lq = np.zeros(4)
        mujoco.mju_mulQuat(lq, biq, gq)
        typ = {int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
               int(mujoco.mjtGeom.mjGEOM_BOX): "box"}[int(mc.geom_type[g])]
        size = mc.geom_size[g][: {"capsule": 2, "sphere": 1, "box": 3}[typ]]
        el = ET.Element("geom", {"name": mc.geom(g).name, "type": typ, "size": _s(size), "pos": _s(lp),
                                 "quat": _s(lq), "group": "3", "from": src})
        bodies[dst].append(el)
        n += 1
    return n


def add_drive(root: ET.Element) -> None:
    """Steps 3 and 4: gravity compensation, armature and friction."""
    for b in root.iter("body"):
        if b.get("name") != "base":
            b.set("gravcomp", "1")
    for j in root.iter("joint"):
        name = j.get("name")
        kind = MOTORS.get(name, "finger" if j.get("type") == "slide" else None)
        if kind:
            j.set("armature", f"{ARMATURE[kind]:g}")
            j.set("frictionloss", f"{FRICTIONLOSS:g}")


def build(i2rt: Path) -> str:
    root = merge(i2rt)
    root.set("model", "i2rt_yam_v1_linear_4310")
    mi = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    lift = shoulder_lift(mi, mujoco.MjModel.from_xml_path(str(CANONICAL_XML)))
    add_collision(root, lift)
    add_drive(root)
    for g in root.iter("geom"):      # ``from`` is a note for this script's check, not MJCF
        g.attrib.pop("from", None)
    ET.indent(root, "  ")
    try:
        commit = subprocess.run(["git", "-C", str(i2rt), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    digests = {p: hashlib.sha256((i2rt / p).read_bytes()).hexdigest()[:16] for p in (ARM, GRIPPER, MOUNT)}
    head = ("<!--\nGenerated by sim/yam_pack/models/build_i2rt.py from i2rt robotics' YAM release.\n"
            f"i2rt commit: {commit}\n" + "".join(f"{p}: sha256 {h}...\n" for p, h in digests.items()) +
            "Arm yam/v1 + gripper linear_4310, merged as i2rt's combine_arm_and_gripper_xml does. Meshes dropped;\n"
            "collision capsules from the canonical model (yam_collision.xml) on the matching bodies,\n"
            f"lifted {lift[2] * 1000:.2f} mm above joint 1; gravcomp on moving bodies; armature and frictionloss\n"
            "from the canonical model's DM4340/DM4310 classes. Kinematics, ranges, inertials: i2rt's.\n"
            "License: MIT, I2RT Robotics (I2RT_LICENSE).\n-->\n")
    xml = ET.tostring(root, encoding="unicode")
    xml = re.sub(r" />", "/>", xml)
    return head + xml + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("i2rt", type=Path, help="an i2rt checkout (github.com/i2rt-robotics/i2rt)")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(args.i2rt))
    (OUT_DIR / "I2RT_LICENSE").write_text((args.i2rt / "LICENSE").read_text())
    print(f"wrote {OUT.relative_to(HERE.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
