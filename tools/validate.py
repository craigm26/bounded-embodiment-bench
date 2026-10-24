"""Validate the benchmark's declarations. Every embodiment envelope, and every task variant's
envelope with its patch applied, must load with the released Bounded Embodiment loader
(`bounded_embodiment.envelope.load`, envelope 0.2), which runs the schema and the loader's own
semantic checks. Each zone a task adds must sit, in x-y, inside a keep_in zone for the same points.

    python3 -m tools.validate        # exit 0 when everything is consistent
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml
from bounded_embodiment.envelope import EnvelopeError, load

from tools.checks import CHECKS

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {"bimanual": "abc-bimanual", "single_arm": "so-arm101"}
GROUPS = ["work_area", "speed_people", "stopping", "forbidden_authority", "handover"]
TASK_COUNT = 20


def load_embodiment(name: str) -> dict:
    d = ROOT / "embodiments" / name
    meta = yaml.safe_load((d / "embodiment.yaml").read_text())
    doc = json.loads((d / meta["envelope"]).read_text())
    tools = [p["id"] for p in doc["points"] if p["id"].endswith("tool")]
    if len(tools) != meta["arms"]:
        raise ValueError(f"{name}: arms={meta['arms']} but {len(tools)} tool points")
    return {"meta": meta, "envelope": doc}


def envelope_errors(doc: dict) -> list[str]:
    try:
        load(copy.deepcopy(doc))
    except EnvelopeError as exc:
        return [str(exc)]
    return []


def apply_patch(doc: dict, patch: dict | None) -> dict:
    env = copy.deepcopy(doc)
    env["workspace"]["zones"] += copy.deepcopy((patch or {}).get("add_zones", []))
    return env


def _footprint(zone: dict):
    if "box" in zone:
        lo, hi = zone["box"]["min_m"], zone["box"]["max_m"]
        return lo[0], lo[1], hi[0], hi[1]
    xs = [p[0] for p in zone["prism"]["polygon"]]
    ys = [p[1] for p in zone["prism"]["polygon"]]
    return min(xs), min(ys), max(xs), max(ys)


def _covers(keep_in: dict, added: dict, point_ids: list[str]) -> bool:
    applies = keep_in["applies_to"]
    targets = point_ids if added["applies_to"] == ["*"] else added["applies_to"]
    if "*" not in applies and not set(targets) & set(applies):
        return False
    kx0, ky0, kx1, ky1 = _footprint(keep_in)
    x0, y0, x1, y1 = _footprint(added)
    return kx0 <= x0 and ky0 <= y0 and x1 <= kx1 and y1 <= ky1


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
        problems += [f"{name}: {e}" for e in envelope_errors(bodies[variant]["envelope"])]

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
            doc = bodies[variant]["envelope"]
            patched = apply_patch(doc, r.get("envelope_patch"))
            problems += [f"{where}: {e}" for e in envelope_errors(patched)]
            point_ids = [p["id"] for p in doc["points"]]
            keep_ins = [z for z in doc["workspace"]["zones"] if z["kind"] == "keep_in"]
            for z in (r.get("envelope_patch") or {}).get("add_zones", []):
                if z.get("kind") == "keep_out" and not any(_covers(k, z, point_ids) for k in keep_ins):
                    problems.append(f"{where}: zone {z.get('id')} leaves keep_in")
    return problems


def main() -> int:
    problems = validate()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"ok: {TASK_COUNT} tasks x {len(VARIANTS)} embodiments, all envelopes load as 0.2")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
