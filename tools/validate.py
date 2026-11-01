"""Validate the benchmark's declarations. tasks.yaml is loaded strictly first: a duplicate key is an
error, and the result must match tasks/tasks.schema.json, where every object is closed and every text
field is a non-empty string (an unquoted comma in a flow mapping turns half a value into a stray key
with a null value; this catches it). Every embodiment envelope, and every task variant's envelope with
its patch applied, must load with the released Bounded Embodiment loader
(`bounded_embodiment.envelope.load`, envelope 0.2), which runs the schema and the loader's own
semantic checks. Each zone a task adds must sit, in x-y, inside a keep_in zone for the same points.
Every envelope must also be able to stop from its own top speed within its declared stop time and
distance, braking no harder than its acceleration limit (``stopping_problems``).

    python3 -m tools.validate        # exit 0 when everything is consistent
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml
from bounded_embodiment.envelope import EnvelopeError, load
from jsonschema import Draft202012Validator

from tools.checks import CHECKS

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {"bimanual": "abc-bimanual", "single_arm": "so-arm101"}
#: Variants a task MAY have (only the groups that carry over to that embodiment).
OPTIONAL_VARIANTS = {"microduck": "microduck"}
ALL_VARIANTS = {**VARIANTS, **OPTIONAL_VARIANTS}
GROUPS = ["work_area", "speed_people", "stopping", "forbidden_authority", "handover"]
TASK_COUNT = 20
SCHEMA = ROOT / "tasks/tasks.schema.json"


class StrictLoader(yaml.SafeLoader):
    """safe_load, except a mapping that repeats a key is an error instead of keeping the last value."""


def _no_duplicates(loader: StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen = {}
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r} (first at line {seen[key] + 1})", key_node.start_mark)
        seen[key] = key_node.start_mark.line
    return loader.construct_mapping(node, deep=deep)


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def load_tasks(path: Path) -> tuple[dict | None, list[str]]:
    """Load tasks.yaml strictly and check it against the schema. Returns (data, problems); data is None
    when the file does not parse."""
    try:
        data = yaml.load(path.read_text(), Loader=StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        at = f" line {mark.line + 1}" if mark else ""
        return None, [f"tasks.yaml{at}: {getattr(exc, 'problem', None) or exc}"]
    schema = json.loads(SCHEMA.read_text())
    schema["$defs"]["check"]["enum"] = sorted(CHECKS)
    problems = []
    for err in sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.absolute_path)):
        where = _where(data, list(err.absolute_path))
        problems.append(f"schema {where}: {err.message}")
    return data, problems


def _where(data, path: list) -> str:
    """A readable location: the task id in place of its list index."""
    out, node = [], data
    for part in path:
        if isinstance(node, list) and isinstance(part, int) and isinstance(node[part], dict) and "id" in node[part]:
            out.append(str(node[part]["id"]))
        else:
            out.append(str(part))
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            node = None
    return ".".join(out[1:] if out[:1] == ["tasks"] else out) or "(top)"


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


#: Embodiments whose declared stop is known to be shorter than braking allows, left so on purpose. A gap
#: listed here is not reported, and validate fails once it is closed, so the entry has to go with it.
KNOWN_STOPPING_GAPS = {
    "so-arm101": "the same numbers the ABC envelope had (0.25 m/s, 0.5 m/s^2, a 300 ms and 0.03 m stop); its "
                 "envelope is part of every single_arm model-track prompt, so changing it means rerunning the "
                 "model track",
}


def top_speed_mps(doc: dict) -> float:
    """The fastest linear speed the envelope lets any point move: each point's own limit, or the motion
    limit for a point that declares none. Proximity caps are lower by the loader's rules."""
    default = doc["motion"]["max_speed_mps"]
    return max((p.get("max_speed_mps", default) for p in doc["points"]), default=default)


def stopping_problems(doc: dict) -> list[str]:
    """A declared stop must be reachable from the envelope's own top speed.

    Envelope 0.2 declares ``stop.max_time_ms`` and ``max_distance_m`` as the worst case from a stop request
    to standstill at ``max_speed_mps``. Here ``motion.max_accel_mps2`` binds the tools in every frame, a
    stop included (sim/yam_pack/oracle.py), as the reference kit's rover brakes at its own acceleration
    limit. So from speed v the quickest stop takes v / a seconds and v^2 / 2a metres, with no reaction
    time at all. A declaration shorter than that cannot be met by a robot that keeps to the rest of its
    envelope: using its top speed, it either brakes harder than allowed or misses the stop.

    The bound is a necessary condition, not a margin: a real stop adds detection and command latency
    on top. Category 0 sources (power cut) are exempt, since without power the motion limits no longer
    govern how the arm comes to rest. An envelope that declares no acceleration limit gives no bound.
    """
    a = doc.get("motion", {}).get("max_accel_mps2")
    stop = doc.get("stop")
    if not a or not stop:
        return []
    v = top_speed_mps(doc)
    t_min, d_min = v / a, v * v / (2 * a)
    out = []
    for where, s in [("stop", stop)] + [(f"stop.sources[{s['source']}]", s) for s in stop.get("sources", [])]:
        if s.get("category") == 0:
            continue
        if s["max_time_ms"] / 1000.0 < t_min - 1e-9:
            out.append(f"{where}: max_time_ms {s['max_time_ms']} is shorter than braking from the top speed "
                       f"{v} m/s at max_accel_mps2 {a} ({t_min * 1000:.0f} ms)")
        if s["max_distance_m"] < d_min - 1e-9:
            out.append(f"{where}: max_distance_m {s['max_distance_m']} is shorter than braking from the top "
                       f"speed {v} m/s at max_accel_mps2 {a} ({d_min:.4f} m)")
    return out


def apply_patch(doc: dict, patch: dict | None) -> dict:
    """``add_zones`` adds zones; ``handover_speed_mps`` turns every proximity stop rule into a speed
    cap at the same distance, the handover envelope an integrator declares for a handover task
    (envelope 0.2 cannot scope a proximity rule to a zone, so it holds for the whole task)."""
    env = copy.deepcopy(doc)
    patch = patch or {}
    env["workspace"]["zones"] += copy.deepcopy(patch.get("add_zones", []))
    v = patch.get("handover_speed_mps")
    if v is not None:
        env["proximity"] = [{"when": r["when"], "max_speed_mps": v} if r.get("action") == "stop" else r
                            for r in env.get("proximity") or []]
        env["notes"] = (env.get("notes", "") + f" Handover envelope for this task: every proximity stop "
                        f"rule is a {v} m/s cap at the same distance.").strip()
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
        errs = envelope_errors(bodies[variant]["envelope"])
        problems += [f"{name}: {e}" for e in errs]
        # The stopping check reads fields the schema guarantees, so it runs only on a valid envelope.
        bodies[variant]["stopping"] = [] if errs else stopping_problems(bodies[variant]["envelope"])
        if name not in KNOWN_STOPPING_GAPS:
            problems += [f"{name}: {e}" for e in bodies[variant]["stopping"]]
        elif not bodies[variant]["stopping"]:
            problems.append(f"{name}: can now stop from its top speed; remove it from KNOWN_STOPPING_GAPS")

    data, schema_problems = load_tasks(tasks_path)
    problems += schema_problems
    if data is None or not isinstance(data.get("tasks"), list):
        return problems
    tasks = [t for t in data["tasks"] if isinstance(t, dict) and "id" in t]
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
        if not isinstance(t.get("variants"), dict):
            continue  # the schema reported it
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
            for check in (r.get("pass") or []) + ((t.get("twin") or {}).get("pass") or []):
                if check not in CHECKS:
                    problems.append(f"{where}: unknown check {check}")
            if variant not in bodies:
                continue
            doc = bodies[variant]["envelope"]
            patched = apply_patch(doc, r.get("envelope_patch"))
            errs = envelope_errors(patched)
            problems += [f"{where}: {e}" for e in errs]
            # Reported once for the embodiment above; here only what the task's patch adds.
            if not errs:
                problems += [f"{where}: {e}" for e in stopping_problems(patched) if e not in bodies[variant]["stopping"]]
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
        extra = sum(1 for t in load_tasks(ROOT / "tasks/tasks.yaml")[0]["tasks"]
                    for v in OPTIONAL_VARIANTS if v in t["variants"])
        print(f"ok: {TASK_COUNT} tasks x {len(VARIANTS)} embodiments plus {extra} optional variants, "
              "all envelopes load as 0.2")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
