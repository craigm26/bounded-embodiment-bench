# SPDX-FileCopyrightText: 2026 Craig Merry and contributors
# SPDX-License-Identifier: MIT
"""The YAM's declared points in the station frame, from yam_arena's URDF forward kinematics.

Envelope point ids are ``<arm>_<name>``: ``tool`` (the TCP yam_arena plans with), ``wrist`` (link5's origin),
``elbow`` (link4's origin) and the two fingertips ``tip_a`` and ``tip_b``. The arms are not rotated in the station,
so a point's station position is its base-frame position plus the arm's base position (yam_arena.station.BASES),
which is what yam_arena.ik.tcp_pose does for the TCP.
"""

from __future__ import annotations

import numpy as np
from yam_arena import YAM_URDF, station
from yam_arena.constants import ARM_JOINTS, TCP_OFFSET
from yam_arena.ik import LOWER, UPPER
from yam_arena.urdf import forward_kinematics

LINKS = {"wrist": "link5", "elbow": "link4", "tip_a": "tip_left", "tip_b": "tip_right"}
POINTS = ("tool", *LINKS)


def fk(arm: str, q6) -> dict[str, np.ndarray]:
    poses = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q6, strict=True)})
    base = np.asarray(station.BASES[arm], float)
    g = poses["gripper"]
    out = {f"{arm}_tool": g[:3, :3] @ np.asarray(TCP_OFFSET) + g[:3, 3] + base}
    for name, link in LINKS.items():
        out[f"{arm}_{name}"] = poses[link][:3, 3] + base
    return out


JOINT_LOWER, JOINT_UPPER = LOWER, UPPER
