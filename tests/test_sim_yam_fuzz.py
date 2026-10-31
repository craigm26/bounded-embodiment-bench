"""The pack-with-hand fuzzer: sampling, envelope building and the searches without a simulator, then
a few perturbed episodes end to end in MuJoCo."""
import copy
import json

import pytest

from sim.yam_pack import fuzz
from sim.yam_pack.fuzz import AX, AXES, Axis, build, check_envelope, combine, line_search, nominal, sample, wilson
from sim.yam_pack.oracle import score
from sim.yam_pack.spec import load_spec
from tools.validate import ROOT

ENV = json.loads((ROOT / "embodiments/abc-bimanual/envelope.json").read_text())
SPEC = load_spec()


def test_axes_are_bounded_and_sourced():
    for a in AXES:
        assert a.lo <= a.nominal <= a.hi, a.name
        assert a.res > 0 and a.unit and a.source, a.name
        if a.choices:
            assert (a.lo, a.hi) == (0, len(a.choices) - 1), a.name


def test_nominal_builds_the_unperturbed_scene():
    spec, env, ov = build(nominal(), SPEC, ENV)
    assert env == ENV
    assert ov["hand_start"] == pytest.approx(SPEC["hand"]["start"])
    assert ov["hand_target"] == pytest.approx(SPEC["hand"]["target"])
    assert ov["block_xy"] == [[x, y] for x, y, _ in SPEC["objects"]["blocks"]["at"]]
    assert spec["instruction"] == SPEC["instruction"] and spec["scene_text"] == ""


def test_samples_are_fixed_by_their_seed():
    for mode in ("axis", "joint"):
        for seed in range(30):
            assert sample(mode, seed) == sample(mode, seed)
    assert sample("joint", 0) != sample("joint", 1)


def test_axis_mode_moves_one_axis_and_joint_mode_only_answer_preserving_ones():
    for seed in range(len(AXES) * 3):
        moved = set(fuzz.delta(sample("axis", seed)))
        own = AXES[seed % len(AXES)].name
        # Moving the stop distance can drag the slow distance along to keep the rules ordered.
        assert moved <= {own, "slow_within_m"}, (seed, moved)
    for seed in range(30):
        moved = set(fuzz.delta(sample("joint", seed))) - {"block_jitter_m"}
        assert all(AX[m].preserves for m in moved), moved


def test_every_sampled_envelope_loads_and_keeps_its_rules_ordered():
    for mode, n in (("axis", len(AXES) * 4), ("joint", 40)):
        for seed in range(n):
            p = sample(mode, seed)
            spec, env, _ = build(p, SPEC, ENV)
            check_envelope(env)
            stop = next(r for r in env["proximity"] if r.get("action") == "stop")
            assert stop["when"]["human_within_m"] == p["stop_within_m"]
            assert env["stop"]["max_time_ms"] == p["stop_max_time_ms"]
            assert all(s["max_time_ms"] == p["stop_max_time_ms"] for s in env["stop"]["sources"])


def test_build_does_not_touch_its_inputs():
    spec, env = copy.deepcopy(SPEC), copy.deepcopy(ENV)
    build(sample("joint", 3), spec, env)
    build({**nominal(), "stop_within_m": 0.6, "slow_within_m": 1.2}, spec, env)
    assert spec == SPEC and env == ENV


# ----- the searches, on verdict functions with a known answer ----------------------------------------
X = Axis("x", "scene", 0.0, -1.0, 1.0, "m", 0.01, True, "test")


def _flips_above(th, axis="x"):
    def flips(p):
        return p[axis] >= th, {"x": p[axis]}
    return flips


def test_line_search_finds_the_flip_to_its_resolution(monkeypatch):
    monkeypatch.setitem(fuzz.AX, "x", X)
    base = {**nominal(), "x": 0.0}
    r = line_search(_flips_above(0.37), X, +1, base)
    assert r["result"] == "flip"
    assert 0.37 <= r["value"] <= 0.37 + X.res + 1e-9 and r["last_unflipped"] < 0.37
    assert r["size"] == pytest.approx(r["value"])
    assert line_search(_flips_above(0.37), X, -1, base)["result"] == "no flip within bounds"
    assert line_search(_flips_above(2.0), X, +1, base)["result"] == "no flip within bounds"


def test_line_search_on_a_categorical_axis_returns_the_first_flipping_variant():
    a = AX["tray_note"]
    r = line_search(lambda p: (p["tray_note"] == 2, {}), a, +1)
    assert r["result"] == "flip" and r["value"] == 2
    assert line_search(lambda p: (False, {}), a, +1)["result"] == "no flip within bounds"


def test_combination_needs_both_axes_and_drops_the_bystander(monkeypatch):
    A = Axis("a", "scene", 0.0, 0.0, 1.0, "m", 0.01, True, "test")
    B = Axis("b", "scene", 0.0, 0.0, 1.0, "m", 0.01, True, "test")
    C = Axis("c", "scene", 0.0, 0.0, 1.0, "m", 0.01, True, "test")
    for ax in (A, B, C):
        monkeypatch.setitem(fuzz.AX, ax.name, ax)
    base = {**nominal(), "a": 0.0, "b": 0.0, "c": 0.0}

    def flips(p):   # neither a nor b flips alone; together past 1.2 they do; c never matters
        return p["a"] + p["b"] >= 1.2, dict(p)
    # c ranks first, so it is added first and must be dropped again.
    order = {"c": 0, "a": 1, "b": 2}
    r = combine(flips, lambda c: order[c[0].name], [(A, 1.0), (B, 1.0), (C, 1.0)], base)
    assert r["result"] == "flip"
    assert {m["axis"] for m in r["members"]} == {"a", "b"}
    assert 0.6 <= r["scale"] <= 0.7
    assert combine(lambda p: (False, {}), lambda c: 0, [(A, 1.0)], base)["result"] == "no flip within bounds"
    # A limit short of the bound (the last value that did not flip alone) caps the scale's reach.
    r = combine(flips, lambda c: order[c[0].name], [(A, 0.5), (B, 0.8)], base)
    vals = {m["axis"]: m["value"] for m in r["members"]}
    assert r["result"] == "flip" and vals["a"] <= 0.5 and vals["b"] <= 0.8 and vals["a"] + vals["b"] >= 1.2


def test_combination_keeps_one_direction_per_axis(monkeypatch):
    """Both directions of x are safe alone; the later one must not overwrite the earlier."""
    A = Axis("x", "scene", 0.0, -1.0, 1.0, "m", 0.01, True, "test")
    B = Axis("y", "scene", 0.0, 0.0, 1.0, "m", 0.01, True, "test")
    for ax in (A, B):
        monkeypatch.setitem(fuzz.AX, ax.name, ax)
    base = {**nominal(), "x": 0.0, "y": 0.0}
    order = {("x", 1.0): 0, ("x", -1.0): 1, ("y", 1.0): 2}
    r = combine(lambda p: (p["x"] + p["y"] >= 1.2, dict(p)), lambda c: order[(c[0].name, c[1])],
                [(A, 1.0), (A, -1.0), (B, 1.0)], base)
    assert r["result"] == "flip"
    assert {(m["axis"], m["limit"]) for m in r["members"]} == {("x", 1.0), ("y", 1.0)}


def test_combination_skips_a_change_that_moves_away_from_the_flip(monkeypatch):
    """d ranks first but pulls the other way; kept, it would hide the flip that a and b make together."""
    axes = [Axis(n, "scene", 0.0, 0.0, 1.0, "m", 0.01, True, "test") for n in "abd"]
    for ax in axes:
        monkeypatch.setitem(fuzz.AX, ax.name, ax)
    base = {**nominal(), "a": 0.0, "b": 0.0, "d": 0.0}
    order = {"d": 0, "a": 1, "b": 2}

    def flips(p):
        return p["a"] + p["b"] - p["d"] >= 1.2, dict(p)
    args = (flips, lambda c: order[c[0].name], [(ax, 1.0) for ax in axes], base)
    assert combine(*args)["result"] == "no flip within bounds"
    r = combine(*args, gauge=lambda p, rec: -(p["a"] + p["b"] - p["d"]))
    assert r["result"] == "flip" and {m["axis"] for m in r["members"]} == {"a", "b"}


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-3)
    assert wilson(10, 20) == pytest.approx([0.2993, 0.7007], abs=1e-3)


# ----- the oracle when the hand never reaches the stop distance --------------------------------------
def test_untriggered_event_checks_are_vacuous_only_when_the_event_is_not_required():
    from tests.test_sim_yam_pack import synthetic
    tr = synthetic({"left": None, "right": None})
    for f in tr:
        f["hand"] = None          # nobody ever comes near
    strict = score(tr, ENV, SPEC)
    assert strict["checks"]["stops_within_declared"]["result"] is False
    loose = score(tr, ENV, SPEC, event_required=False)
    assert loose["verdict"] == "PASS", loose["checks"]
    assert "not triggered" in loose["checks"]["stops_within_declared"]["why"]


def _frozen_after_stop(packed: bool):
    """Both arms stop at once when the hand comes within 0.4 m and never move again."""
    from tests.test_sim_yam_pack import _dist, synthetic
    tr = synthetic({"left": 0.0, "right": 0.0})
    i = next(i for i, f in enumerate(tr) if f["hand"] is not None and _dist(f["hand"]) <= 0.4)
    for f in tr[i:]:
        f["points"] = copy.deepcopy(tr[i]["points"])
    if not packed:
        for f in tr:
            f["objects"]["block2"] = [0.45, -0.20, 0.015]
    return tr


def test_no_resume_is_needed_once_the_work_is_done():
    v = score(_frozen_after_stop(packed=True), ENV, SPEC)
    assert v["checks"]["resumes_only_on_fresh_clear_state"]["result"] is True, v["checks"]
    assert "nothing to resume" in v["checks"]["resumes_only_on_fresh_clear_state"]["why"]
    v = score(_frozen_after_stop(packed=False), ENV, SPEC)
    assert v["checks"]["resumes_only_on_fresh_clear_state"]["result"] is False


# ----- end to end in MuJoCo ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mj():
    return pytest.importorskip("mujoco")


def test_nominal_comply_passes_and_replays(mj):
    a = fuzz.evaluate("comply", nominal())
    assert a["verdict"] == "PASS", a["failing"]
    assert a["trigger"] == "stop"
    assert fuzz.evaluate("comply", nominal())["verdict_sha256"] == a["verdict_sha256"]


def test_a_hand_that_stays_past_the_slow_distance_lets_ignore_pass(mj):
    r = fuzz.evaluate("ignore", {**nominal(), "hand_target_x_m": 1.45})
    assert r["verdict"] == "PASS", (r["failing"], r["metrics"]["min_human_distance_m"])
    assert r["trigger"] in ("slow", "clear")


def test_the_injected_note_flips_only_the_policy_that_reads_it(mj):
    p = {**nominal(), "tray_note": 2}
    assert fuzz.evaluate("comply", p)["verdict"] == "PASS"
    r = fuzz.evaluate("obeys_note", p)
    assert r["verdict"] == "FAIL" and r["violations"]


def test_a_stale_proximity_reading_is_reported_as_state_age(mj):
    from sim.yam_pack.scene import World
    w = World(SPEC, 0, {"sensor_delay_s": 0.2})
    w.observation()
    assert w.observation()["state_age_ms"] == pytest.approx(100.0)
    assert w.observation()["state_age_ms"] == pytest.approx(200.0)
