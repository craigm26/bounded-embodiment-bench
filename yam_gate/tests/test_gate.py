# SPDX-FileCopyrightText: 2026 Craig Merry and contributors
# SPDX-License-Identifier: MIT
"""Offline tests of the YAM Bounded Embodiment gate: no Isaac, no Dream. Needs bounded_embodiment (core) and a
yam-isaac-lab-arena checkout on YAM_ARENA (default ~/projects/ext/yam-isaac-lab-arena).

    YAM_ARENA=... python -m pytest yam_gate/tests -q      (or python yam_gate/tests/test_gate.py)
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, os.path.expanduser(os.environ.get("YAM_ARENA", "~/projects/ext/yam-isaac-lab-arena")))
sys.dont_write_bytecode = True

from yam_arena import station  # noqa: E402
from yam_arena.constants import READY  # noqa: E402
from yam_arena.demo import ArmPlan, pick_place_waypoints  # noqa: E402

from be_yam_gate.gate import YamEnvelopeGate  # noqa: E402
from be_yam_gate.yam import JOINT_LOWER, JOINT_UPPER, fk  # noqa: E402

ENV = HERE.parent / "envelopes" / "keep-estop-zone-clear.envelope.json"
HZ = 20.0
TOP = 0.7475


def rows_for(plans: dict, seconds: float) -> list[np.ndarray]:
    arms = {a: ArmPlan(list(plans[a]), station.BASES[a]) if a in plans else None for a in ("left", "right")}
    out = []
    for k in range(int(seconds * HZ)):
        row = []
        for a in ("left", "right"):
            if arms[a] is None:
                row += [*READY, 1.0]
            else:
                row += list(arms[a].action7(k / HZ)[0])
        out.append(np.array(row))
    return out


def min_clearance(gate, row) -> float:
    worst = math.inf
    for a, arm in enumerate(("left", "right")):
        pts = fk(arm, row[a * 7:a * 7 + 6])
        for pid in gate.points[arm]:
            worst = min(worst, gate.envelope.clearance(pid, pts[pid]))
    return worst


class GateTests(unittest.TestCase):
    def new_gate(self):
        return YamEnvelopeGate(ENV, fk, JOINT_LOWER, JOINT_UPPER, HZ, margin_m=0.005)

    def test_old_demo_right_arm_is_kept_out(self):
        """PR #2's first demo: the right arm carried the blue cube from beside the zone. Ungated it enters the
        striped area; gated, nothing it is sent ever goes inside a keep-out."""
        cube = (0.23, -0.05, TOP + 0.012 + 0.02)
        right = pick_place_waypoints(cube, (0.36, -0.05, TOP + 0.025), travel_z=TOP + 0.14, yaw=-np.pi / 2)
        rows = rows_for({"right": right}, 12.0)
        ungated_worst = min(min_clearance(self.new_gate(), r) for r in rows)
        self.assertLess(ungated_worst, 0.0, "the ungated plan should violate the envelope")
        g = self.new_gate()
        g.reset(rows[0])
        sent = [g.filter(r) for r in rows[1:]]
        gated_worst = min(min_clearance(g, r) for r in sent)
        self.assertGreaterEqual(gated_worst, 0.0)
        self.assertGreater(g.stats.held + g.stats.cut, 0)

    def test_new_demo_passes_untouched(self):
        """The resubmitted demo (left arm only, away from the zone) needs no intervention for the zone."""
        red, blue = (0.18, 0.12, TOP + 0.032), (0.22, 0.04, TOP + 0.032)
        left = pick_place_waypoints(red, (0.14, 0.23, TOP + 0.025), travel_z=TOP + 0.14, yaw=-np.pi / 2)
        left += pick_place_waypoints(blue, (0.26, 0.23, TOP + 0.025), travel_z=TOP + 0.14, yaw=-np.pi / 2)
        rows = rows_for({"left": left}, 20.0)
        g = self.new_gate()
        g.reset(rows[0])
        for r in rows[1:]:
            g.filter(r)
        self.assertEqual(g.stats.held, 0)
        self.assertEqual(g.stats.cut, 0)

    def test_invalid_row_holds(self):
        g = self.new_gate()
        start = np.array([*READY, 1.0, *READY, 1.0])
        g.reset(start)
        out = g.filter(np.full(14, np.nan))
        np.testing.assert_allclose(out, start)
        self.assertEqual(g.stats.invalid, 1)

    def test_speed_cap(self):
        """A request to jump far in one step is slowed to the tool speed limit."""
        g = self.new_gate()
        start = np.array([*READY, 1.0, *READY, 1.0])
        g.reset(start)
        goal = start.copy()
        goal[0] += 0.5  # left base joint swings 0.5 rad in one 50 ms step
        out = g.filter(goal)
        p0 = fk("left", start[:6])["left_tool"]
        p1 = fk("left", out[:6])["left_tool"]
        self.assertLessEqual(np.linalg.norm(p1 - p0), 0.8 / HZ + 1e-6)

    def test_grippers_pass_through(self):
        g = self.new_gate()
        start = np.array([*READY, 1.0, *READY, 1.0])
        g.reset(start)
        cmd = start.copy()
        cmd[6], cmd[13] = 0.0, 0.3
        out = g.filter(cmd)
        self.assertEqual(out[6], 0.0)
        self.assertEqual(out[13], 0.3)


if __name__ == "__main__":
    unittest.main()
