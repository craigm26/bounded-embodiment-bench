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
#: Variants a task MAY have (only the groups that carry over to that embodiment).
OPTIONAL_VARIANTS = {"microduck": "microduck"}
ALL_VARIANTS = {**VARIANTS, **OPTIONAL_VARIANTS}
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
    xs = [p[0] for p in zone["prism"]["polygon_m"]]
    ys = [p[1] for p in zone["prism"]["polygon_m"]]
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
    for variant, name in ALL_VARIANTS.items():
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
        for variant in [v for v in ALL_VARIANTS if v in t["variants"]]:
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
    problems += scene_problems(tasks, bodies)
    return problems


def scene_problems(tasks: list[dict], bodies: dict, path: Path = ROOT / "tasks/scenes.yaml") -> list[str]:
    """The model track's scenes: each names a real task variant, its start point is inside the
    patched envelope, and every object a goal or target names is in the scene."""
    if not path.exists():
        return []
    from bounded_embodiment.envelope import load as load_env

    doc = yaml.safe_load(path.read_text())
    by_id = {t["id"]: t for t in tasks}
    out = []
    for tid, variants in doc["scenes"].items():
        for variant, scene in variants.items():
            where = f"scene {tid}[{variant}]"
            if tid not in by_id or variant not in by_id[tid]["variants"]:
                out.append(f"{where}: no such task variant")
                continue
            if variant not in bodies:
                continue
            s = {**doc["defaults"][variant], **scene}
            objs = s.get("objects", {})
            names = [x for pair in (s.get("goal") or {}).get("place", []) for x in pair]
            names += [x for x in (s.get("goal") or {}).get("visit", [])]
            for key in ("target",):
                if s.get(key):
                    names.append(s[key])
            for g in ("seat", "reach"):
                if (s.get("goal") or {}).get(g):
                    names.append(s["goal"][g]["at"])
            out += [f"{where}: unknown object {n}" for n in names if n not in objs]
            try:
                env = load_env(apply_patch(bodies[variant]["envelope"],
                                           resolve(by_id[tid], variant).get("envelope_patch")))
            except EnvelopeError:
                continue  # reported above, with the task variant's envelope
            c = env.clearance(s["point"], s["start"]["at"])
            if c <= 0:
                out.append(f"{where}: start point outside the envelope (clearance {c:.3f} m)")
    return out


def main() -> int:
    problems = validate()
    for p in problems:
        print("FAIL", p)
    if not problems:
        extra = sum(1 for t in yaml.safe_load((ROOT / "tasks/tasks.yaml").read_text())["tasks"]
                    for v in OPTIONAL_VARIANTS if v in t["variants"])
        print(f"ok: {TASK_COUNT} tasks x {len(VARIANTS)} embodiments plus {extra} optional variants, "
              "all envelopes load as 0.2")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
