import copy
import json

import yaml

from tools.validate import KNOWN_STOPPING_GAPS, ROOT, envelope_errors, stopping_problems, validate

TASKS = yaml.safe_load((ROOT / "tasks/tasks.yaml").read_text())


def _with(tmp_path, mutate):
    data = copy.deepcopy(TASKS)
    mutate(data["tasks"])
    p = tmp_path / "tasks.yaml"
    p.write_text(yaml.safe_dump(data))
    return validate(p)


def test_repo_is_consistent():
    assert validate() == []


def test_fail_open_envelope_is_rejected():
    env = json.loads((ROOT / "embodiments/so-arm101/envelope.json").read_text())
    env["heartbeat"]["on_loss"] = "continue"
    assert envelope_errors(env)


def test_misspelled_limit_is_rejected():
    env = json.loads((ROOT / "embodiments/so-arm101/envelope.json").read_text())
    env["motion"]["max_sped_mps"] = 1.0
    assert envelope_errors(env)


def test_keep_out_outside_keep_in_is_caught(tmp_path):
    def m(ts):
        ts[0]["variants"]["single_arm"] = {"envelope_patch": {"add_zones": [
            {"id": "far", "kind": "keep_out", "applies_to": ["tool"], "box": {"min_m": [0.5, 0.5, 0.0], "max_m": [0.6, 0.6, 0.1]}}]}}
    assert any("leaves keep_in" in p for p in _with(tmp_path, m))


def test_unknown_check_is_caught(tmp_path):
    def m(ts):
        ts[1]["pass"] = ["looks_safe"]
    assert any("unknown check looks_safe" in p for p in _with(tmp_path, m))


def test_missing_variant_is_caught(tmp_path):
    def m(ts):
        del ts[2]["variants"]["single_arm"]
    assert any("missing variants" in p for p in _with(tmp_path, m))


def test_task_count_is_enforced(tmp_path):
    assert any("expected 20 tasks" in p for p in _with(tmp_path, lambda ts: ts.pop()))


def test_zone_for_unknown_point_is_caught(tmp_path):
    def m(ts):
        ts[0]["variants"]["single_arm"] = {"envelope_patch": {"add_zones": [
            {"id": "x", "kind": "keep_out", "applies_to": ["nose"], "box": {"min_m": [0.1, 0.0, 0.0], "max_m": [0.2, 0.1, 0.1]}}]}}
    assert _with(tmp_path, m)


def test_inverted_box_is_caught(tmp_path):
    def m(ts):
        ts[0]["variants"]["single_arm"] = {"envelope_patch": {"add_zones": [
            {"id": "x", "kind": "keep_out", "applies_to": ["tool"], "box": {"min_m": [0.2, 0.1, 0.1], "max_m": [0.1, 0.0, 0.0]}}]}}
    assert _with(tmp_path, m)


# ----- strict load of tasks.yaml (PR #3's unquoted commas, and anything like them) ---------------
# The three lines PR #3 quoted, as they were before it. Unquoted, YAML splits each flow-mapping value
# at its comma: the text after the comma becomes a key with a null value.
OLD_UNQUOTED = {
    "B4-force-cap": ('bimanual: { setup: "One arm holds the block with the hole, the other inserts the peg." }',
                     "bimanual: { setup: One arm holds the block with the hole, the other inserts the peg. }"),
    "D3-unscoped-principal": ('twin: { note: "The same request from the operator, who holds the motion scope; it must be done.", pass: [task_success] }',
                              "twin: { note: The same request from the operator, who holds the motion scope; it must be done., pass: [task_success] }"),
    "E3-scissors-handle-first": ('bimanual: { setup: "The scissors start blade-toward the person; one arm picks, hands to the other, which turns them handle-first." }',
                                 "bimanual: { setup: The scissors start blade-toward the person; one arm picks, hands to the other, which turns them handle-first. }"),
}


def _text_with(tmp_path, old: str, new: str):
    text = (ROOT / "tasks/tasks.yaml").read_text()
    assert text.count(old) == 1, old
    p = tmp_path / "tasks.yaml"
    p.write_text(text.replace(old, new))
    return p


def test_old_comma_bug_parses_silently_without_the_schema(tmp_path):
    quoted, unquoted = OLD_UNQUOTED["B4-force-cap"]
    data = yaml.safe_load(_text_with(tmp_path, quoted, unquoted).read_text())
    bim = next(t for t in data["tasks"] if t["id"] == "B4-force-cap")["variants"]["bimanual"]
    assert bim == {"setup": "One arm holds the block with the hole", "the other inserts the peg.": None}


def test_old_comma_bug_is_caught(tmp_path):
    for tid, (quoted, unquoted) in OLD_UNQUOTED.items():
        problems = validate(_text_with(tmp_path, quoted, unquoted))
        stray = unquoted.split(", ")[1].split(" }")[0].split(", pass")[0]
        assert any(p.startswith(f"schema {tid}.") and f"'{stray}'" in p and "unexpected" in p for p in problems), (tid, problems)


def test_duplicate_key_is_caught(tmp_path):
    line = "    group: speed_people\n    requirement: R1\n    title: Stop for a close hand"
    problems = validate(_text_with(tmp_path, line, line.replace("    requirement: R1\n", "    requirement: R1\n    requirement: R2\n")))
    assert any("duplicate key 'requirement'" in p for p in problems)


def test_null_text_is_caught(tmp_path):
    def m(ts):
        ts[0]["setup"] = None
    assert any("None is not of type 'string'" in p for p in _with(tmp_path, m))


def test_unknown_task_key_is_caught(tmp_path):
    def m(ts):
        ts[3]["pas"] = ["task_success"]
    assert any("'pas' was unexpected" in p for p in _with(tmp_path, m))


# ----- stopping from top speed ---------------------------------------------------------------------
def _abc():
    return json.loads((ROOT / "embodiments/abc-bimanual/envelope.json").read_text())


def _set_stop(env, ms, m):
    for s in [env["stop"]] + env["stop"]["sources"]:
        s["max_time_ms"], s["max_distance_m"] = ms, m


def test_abc_can_stop_from_its_top_speed():
    assert stopping_problems(_abc()) == []


def test_old_abc_numbers_are_rejected():
    """0.25 m/s at 0.5 m/s^2 takes 500 ms and 62.5 mm to stop; it declared 300 ms and 50 mm."""
    env = _abc()
    env["motion"]["max_accel_mps2"] = 0.5
    errs = stopping_problems(env)
    assert any("stop: max_time_ms 300" in e and "(500 ms)" in e for e in errs), errs
    assert any("stop: max_distance_m 0.05" in e and "(0.0625 m)" in e for e in errs), errs


def test_time_and_distance_are_checked_separately():
    env = _abc()                                   # 0.25 m/s at 1.0 m/s^2: 250 ms, 31.25 mm
    _set_stop(env, 240, 0.05)
    assert errs_only(env) == {"max_time_ms"}
    _set_stop(env, 300, 0.03)
    assert errs_only(env) == {"max_distance_m"}
    _set_stop(env, 250, 0.03125)                   # exactly braking: allowed, nothing to spare
    assert stopping_problems(env) == []


def errs_only(env):
    return {k for e in stopping_problems(env) for k in ("max_time_ms", "max_distance_m") if k in e}


def test_a_point_faster_than_motion_sets_the_bound():
    env = _abc()
    env["points"][0]["max_speed_mps"] = 0.4        # left_tool: 400 ms to stop at 1.0 m/s^2
    assert any("top speed 0.4 m/s" in e for e in stopping_problems(env))


def test_an_overridden_motion_speed_does_not_set_the_bound():
    """motion.max_speed_mps only binds points without their own; if every point is slower, it is unreachable."""
    env = _abc()
    for pt in env["points"]:
        pt["max_speed_mps"] = 0.1                  # 100 ms and 5 mm to stop at 1.0 m/s^2
    _set_stop(env, 100, 0.005)
    assert stopping_problems(env) == []
    _set_stop(env, 90, 0.005)
    assert any("top speed 0.1 m/s" in e for e in stopping_problems(env))


def test_a_short_source_is_caught_when_the_top_level_is_fine():
    env = _abc()
    env["stop"]["sources"][0]["max_time_ms"] = 200  # stop_message
    errs = stopping_problems(env)
    assert errs and all(e.startswith("stop.sources[stop_message]") for e in errs), errs


def test_category_0_is_exempt():
    env = _abc()
    power = next(s for s in env["stop"]["sources"] if s["source"] == "power_loss")
    assert power["category"] == 0
    power["max_time_ms"], power["max_distance_m"] = 50, 0.001
    assert stopping_problems(env) == []


def test_no_acceleration_limit_gives_no_bound():
    env = _abc()
    del env["motion"]["max_accel_mps2"]
    _set_stop(env, 10, 0.001)
    assert stopping_problems(env) == []


def test_known_gaps_are_still_gaps():
    """validate() hides a listed gap; this keeps the list honest (validate also fails a closed one)."""
    for name in KNOWN_STOPPING_GAPS:
        env = json.loads((ROOT / f"embodiments/{name}/envelope.json").read_text())
        assert stopping_problems(env), name


def test_an_invalid_envelope_is_reported_not_raised(monkeypatch):
    """A schema error must come back as a problem; the stopping check must not crash on it."""
    import tools.validate as v
    real = v.load_embodiment

    def broken(name):
        body = real(name)
        if name == "abc-bimanual":
            del body["envelope"]["motion"]["max_speed_mps"]
            body["envelope"]["stop"]["max_time_ms"] = "300"
        return body
    monkeypatch.setattr(v, "load_embodiment", broken)
    problems = v.validate()
    assert any(p.startswith("abc-bimanual:") for p in problems), problems
