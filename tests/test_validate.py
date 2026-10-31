import copy
import json

import yaml

from tools.validate import ROOT, envelope_errors, validate

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
