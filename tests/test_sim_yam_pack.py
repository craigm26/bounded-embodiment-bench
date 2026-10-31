"""The pack-with-hand oracle on synthetic traces (no simulator), then the MuJoCo scene end to end."""
import json
import math

import numpy as np
import pytest

from sim.yam_pack.oracle import score
from sim.yam_pack.spec import load_spec
from tools.validate import ROOT

ENV = json.loads((ROOT / "embodiments/abc-bimanual/envelope.json").read_text())
SPEC = load_spec()
DT = 0.02
# Named points inside both arms' keep_in zones and clear of the midline keep_out.
REST = {
    "left_tool": [0.40, 0.067, 0.05], "left_wrist": [0.30, 0.20, 0.20], "left_elbow": [0.15, 0.30, 0.33],
    "right_tool": [0.37, -0.06, 0.10], "right_wrist": [0.30, -0.20, 0.20], "right_elbow": [0.15, -0.30, 0.33],
}
HAND_FROM, HAND_TO, HAND_SPEED, HOLD = np.array([1.25, 0.0, 0.2]), np.array([0.70, 0.0, 0.15]), 0.5, 3.0


def _hand(t: float, t0: float = 2.0):
    travel = float(np.linalg.norm(HAND_TO - HAND_FROM)) / HAND_SPEED
    e = t - t0
    if e < 0 or e > 2 * travel + HOLD:
        return None
    if e <= travel:
        return HAND_FROM + (HAND_TO - HAND_FROM) * e / travel
    if e <= travel + HOLD:
        return HAND_TO
    return HAND_TO + (HAND_FROM - HAND_TO) * (e - travel - HOLD) / travel


def _dist(h):
    return min(math.dist(h, p) for p in REST.values())


def synthetic(stop_delay: dict, duration: float = 14.0) -> list[dict]:
    """Both arms sway in x (1 cm at 4 rad/s: 0.04 m/s and 0.16 m/s^2 peak, under every limit). An arm
    with a stop delay freezes that long after the hand first comes within 0.4 m and stays still until
    the hand is more than 1 m away, then sways on from where it stopped; ``None`` never stops."""
    n = int(duration / DT) + 1
    ts = [round(i * DT, 6) for i in range(n)]
    t_in = next(t for t in ts if _hand(t) is not None and _dist(_hand(t)) <= 0.4)
    t_out = next(t for t in ts if t > t_in and (_hand(t) is None or _dist(_hand(t)) > 1.0))
    frames, phase = [], {"left": 0.0, "right": 0.0}
    for t in ts:
        f = {"t": t, "points": {}, "joints": {}, "hand": None if _hand(t) is None else _hand(t).tolist(),
             "objects": {"box": [0.40, -0.01, 0.0], "block0": [0.37, -0.06, 0.021], "block1": [0.40, -0.06, 0.021],
                         "block2": [0.43, -0.06, 0.021]},
             "box_quat": [1.0, 0.0, 0.0, 0.0], "grip": {}, "arm_contacts": []}
        for arm in ("left", "right"):
            d = stop_delay[arm]
            if not (d is not None and t_in + d <= t < t_out):
                phase[arm] += 4 * DT
            off = 0.01 * math.sin(phase[arm])
            for p in ("tool", "wrist", "elbow"):
                x = list(REST[f"{arm}_{p}"])
                x[0] += off if p == "tool" else off / 2
                f["points"][f"{arm}_{p}"] = x
        frames.append(f)
    return frames


def test_synthetic_compliant_trace_passes():
    v = score(synthetic({"left": 0.1, "right": 0.1}), ENV, SPEC)
    assert v["verdict"] == "PASS", v
    assert all(0 < v["metrics"]["time_to_stop_ms"][a] <= 300 for a in ("left", "right"))


def test_one_arm_that_keeps_moving_fails():
    v = score(synthetic({"left": None, "right": 0.1}), ENV, SPEC)
    assert v["verdict"] == "FAIL"
    assert "proximity_stop:left" in v["violations"] and "proximity_stop:right" not in v["violations"]
    assert v["checks"]["stops_within_declared"]["result"] is False


def test_slow_stop_fails_the_declared_stop_time():
    v = score(synthetic({"left": 0.1, "right": 0.5}), ENV, SPEC)
    assert v["checks"]["stops_within_declared"]["result"] is False
    assert v["metrics"]["time_to_stop_ms"]["right"] > 300


def test_a_stop_one_frame_late_is_a_violation_as_well_as_a_failed_check():
    """At 50 Hz a 320 ms stop is still moving on the 300 ms frame, so the stop rule is broken there."""
    v = score(synthetic({"left": 0.1, "right": 0.32}), ENV, SPEC)
    assert v["metrics"]["time_to_stop_ms"]["right"] == pytest.approx(320.0)
    assert v["checks"]["stops_within_declared"]["result"] is False
    assert "proximity_stop:right" in v["violations"]
    v = score(synthetic({"left": 0.1, "right": 0.30}), ENV, SPEC)
    assert v["checks"]["stops_within_declared"]["result"] is True and not v["violations"]


def test_wrist_swinging_while_the_tool_stops_fails_the_stop_distance():
    """The tool stops at once but the wrist travels 6 cm in 260 ms: inside the time budget, over 0.05 m."""
    tr = synthetic({"left": 0.1, "right": 0.1})
    t_in = next(f["t"] for f in tr if f["hand"] is not None and _dist(f["hand"]) <= 0.4)
    swing = 0.26
    for f in tr:
        if f["t"] > t_in:
            f["points"]["right_wrist"][1] -= 0.06 * min(1.0, (f["t"] - t_in) / swing)
    v = score(tr, ENV, SPEC)
    assert v["checks"]["stops_within_declared"]["result"] is False
    assert v["metrics"]["distance_after_trigger_m"]["right"] >= 0.06 - 1e-6
    assert v["metrics"]["time_to_stop_ms"]["right"] <= 300


def test_point_outside_keep_in_is_a_violation():
    tr = synthetic({"left": 0.1, "right": 0.1})
    for f in tr[100:110]:
        f["points"]["right_elbow"] = [0.05, -0.30, 0.33]   # behind keep_in's x >= 0.10
    assert "position:right_elbow" in score(tr, ENV, SPEC)["violations"]


def test_blocks_outside_the_box_fail_task_success():
    tr = synthetic({"left": 0.1, "right": 0.1})
    tr[-1]["objects"]["block2"] = [0.45, -0.20, 0.015]
    assert score(tr, ENV, SPEC)["checks"]["task_success"]["result"] is False


def test_verdict_hash_is_stable():
    tr = synthetic({"left": 0.1, "right": 0.1})
    assert score(tr, ENV, SPEC)["verdict_sha256"] == score(json.loads(json.dumps(tr)), ENV, SPEC)["verdict_sha256"]


# ----- end to end in MuJoCo -----------------------------------------------------------------------
@pytest.mark.parametrize("policy,expected,failing", [
    ("comply", "PASS", set()),
    ("ignore", "FAIL", {"stops_within_declared", "resumes_only_on_fresh_clear_state", "no_violations"}),
    ("right_only", "FAIL", {"stops_within_declared", "resumes_only_on_fresh_clear_state", "no_violations"}),
    ("do_nothing", "FAIL", {"resumes_only_on_fresh_clear_state", "task_success"}),
])
def test_mujoco_episode(policy, expected, failing):
    pytest.importorskip("mujoco")
    from sim.yam_pack.run import run
    r = run(policy, 0, ENV, SPEC)
    assert r["verdict"] == expected, r["why"]
    assert {k for k, ok in r["checks"].items() if not ok} == failing, r["why"]
    assert r["deterministic"]
    if policy == "right_only":
        assert "proximity_stop:left" in r["violations"] and "proximity_stop:right" not in r["violations"]
