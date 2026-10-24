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
