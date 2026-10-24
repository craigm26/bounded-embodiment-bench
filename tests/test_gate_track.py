"""The gate track, with tiny budgets (the Pi reboots under load: one sim at a time, small runs).

Needs the kit with `bounded_embodiment_kit.falsify` (bounded-embodiment PR #17) and that repo's
checkout at BE_REPO; skipped, with the reason, without them."""
import copy
import json

import pytest

pytest.importorskip("bounded_embodiment_kit.falsify", reason="needs the kit from PR #17")

from tools import gate_track as G  # noqa: E402

if not (G.BE_REPO / "examples/reference-gates").is_dir():
    pytest.skip(f"no bounded-embodiment checkout at {G.BE_REPO}", allow_module_level=True)

TINY = 60


def test_one_task_end_to_end(tmp_path):
    rows = G.run(only=["A2-forbidden-fetch"], budget=TINY, out=tmp_path, control=False)
    by = {r["variant"]: r for r in rows}
    run = by["single_arm"]
    assert run["status"] == "run"
    assert run["adapter_mapping"]["link_map"] == G.SIMULATORS["so-arm101"]["link_map"]
    assert run["preflight"]["start_clear_of_margin"]
    assert [x["property"] for x in run["runs"]] == list(G.PROPERTIES)
    for x in run["runs"]:
        assert x["seed"] == G.SEED and x["budget_steps"] == TINY and x["steps_used"] <= TINY
        assert isinstance(x["found"], bool)
    written = json.loads((tmp_path / "A2-forbidden-fetch-single_arm.json").read_text())
    assert written["status"] == "run"
    # the trace replays bit for bit on a fresh machine and gate
    from bounded_embodiment_kit.falsify import replay

    tr = json.loads((G.ROOT / run["runs"][0]["trace"]).read_text())  # absolute stays absolute
    assert replay(tr, root=G.ROOT)["matches"]


def test_not_run_variants_are_reported_not_dropped(tmp_path):
    rows = G.run(only=["C1-stop-mid-reach"], budget=TINY, out=tmp_path, control=False,
                 properties=("position",))
    assert {r["variant"] for r in rows} == {"bimanual", "single_arm"}
    bi = next(r for r in rows if r["variant"] == "bimanual")
    assert bi["status"] == "not_run" and "no simulator adapter" in bi["reason"]
    assert json.loads((tmp_path / "C1-stop-mid-reach-bimanual.json").read_text())["status"] == "not_run"


def test_start_inside_a_zone_is_not_run_without_an_override(tmp_path, monkeypatch):
    sim = copy.deepcopy(G.SIMULATORS["so-arm101"])
    sim["start_overrides"] = {}
    monkeypatch.setitem(G.SIMULATORS, "so-arm101", sim)
    rows = G.run(only=["E4-cross-the-strip"], budget=TINY, out=tmp_path, control=False)
    arm = next(r for r in rows if r["variant"] == "single_arm")
    assert arm["status"] == "not_run" and "start pose" in arm["reason"]
    assert not arm["preflight"]["start_inside"]


def test_an_unmapped_link_fails_loudly():
    env = json.loads((G.ROOT / "embodiments/so-arm101/envelope.json").read_text())
    env["points"][0]["link"] = "claw"
    with pytest.raises(ValueError, match="link_map"):
        G.map_envelope(env, G.SIMULATORS["so-arm101"])
