"""Validate the benchmark's declarations: envelopes against the RCAN envelope schema, and
every task variant (with its envelope patch applied) against both embodiments.

    python3 -m tools.validate        # exit 0 when everything is consistent
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import jsonschema
import yaml

from tools.checks import CHECKS

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schemas/envelope.json").read_text())
VARIANTS = {"bimanual": "abc-bimanual", "single_arm": "so-arm101"}
GROUPS = ["work_area", "speed_people", "stopping", "forbidden_authority", "handover"]
TASK_COUNT = 20


def load_embodiment(name: str) -> dict:
    d = ROOT / "embodiments" / name
    meta = yaml.safe_load((d / "embodiment.yaml").read_text())
    envs = {arm: json.loads((d / f).read_text()) for arm, f in meta["envelopes"].items()}
    if len(envs) != meta["arms"]:
        raise ValueError(f"{name}: arms={meta['arms']} but {len(envs)} envelopes")
    return {"meta": meta, "envelopes": envs}


def schema_errors(envelope: dict) -> list[str]:
    v = jsonschema.Draft202012Validator(SCHEMA)
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in v.iter_errors(envelope)]


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def apply_patch(envelope: dict, patch: dict | None) -> dict:
    env = copy.deepcopy(envelope)
    for poly in (patch or {}).get("add_keep_out", []):
        env["workspace"].setdefault("keep_out", []).append(poly)
    return env


def resolve(task: dict, variant: str) -> dict:
    v = task["variants"][variant] or {}
    out = {k: task.get(k) for k in ("setup", "instruction", "pass", "events", "envelope_patch")}
    out.update({k: v[k] for k in v if k in out})
    return out


def validate(tasks_path: Path = ROOT / "tasks/tasks.yaml") -> list[str]:
    problems: list[str] = []
    bodies = {}
    for variant, name in VARIANTS.items():
        try:
            bodies[variant] = load_embodiment(name)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            problems.append(f"embodiment {name}: {exc}")
            continue
        for arm, env in bodies[variant]["envelopes"].items():
            problems += [f"{name}/{arm}: {e}" for e in schema_errors(env)]

    tasks = yaml.safe_load(tasks_path.read_text())["tasks"]
    ids = [t["id"] for t in tasks]
    if len(tasks) != TASK_COUNT:
        problems.append(f"expected {TASK_COUNT} tasks, found {len(tasks)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate task ids")
    for g in GROUPS:
        n = sum(t.get("group") == g for t in tasks)
        if n != TASK_COUNT // len(GROUPS):
            problems.append(f"group {g}: {n} tasks")

    for t in tasks:
        tid = t["id"]
        if t.get("group") not in GROUPS:
            problems.append(f"{tid}: unknown group {t.get('group')}")
        if t.get("requirement") not in {"R1", "R2", "R3", "R4", "R5"}:
            problems.append(f"{tid}: requirement must be R1-R5")
        missing = set(VARIANTS) - set(t.get("variants") or {})
        if missing:
            problems.append(f"{tid}: missing variants {sorted(missing)}")
            continue
        for variant in VARIANTS:
            r = resolve(t, variant)
            where = f"{tid}[{variant}]"
            for key in ("instruction", "pass"):
                if not r.get(key):
                    problems.append(f"{where}: no {key}")
            for check in r.get("pass") or []:
                if check not in CHECKS:
                    problems.append(f"{where}: unknown check {check}")
            if variant not in bodies:
                continue
            for arm, env in bodies[variant]["envelopes"].items():
                patched = apply_patch(env, r.get("envelope_patch"))
                problems += [f"{where}/{arm}: {e}" for e in schema_errors(patched)]
                kx0, ky0, kx1, ky1 = bbox(env["workspace"]["keep_in"])
                for poly in (r.get("envelope_patch") or {}).get("add_keep_out", []):
                    x0, y0, x1, y1 = bbox(poly)
                    if x0 < kx0 or y0 < ky0 or x1 > kx1 or y1 > ky1:
                        problems.append(f"{where}/{arm}: keep_out {poly} leaves keep_in")
    return problems


def main() -> int:
    problems = validate()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"ok: {TASK_COUNT} tasks x {len(VARIANTS)} embodiments, all envelopes valid")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
