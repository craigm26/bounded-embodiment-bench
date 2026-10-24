"""The gate track: score the GATE, not the model.

For each task variant whose embodiment has a simulator, build the task's patched envelope
(`tools.validate.apply_patch`), map it onto the simulator's points (an explicit adapter mapping,
written into every result), and run the Bounded Embodiment kit's falsification search
(`be-falsify/1`, draft, simulation only) against the reference arm gate, with a fixed seed and a
small step budget. Writes `results/gate-track/<task>-<variant>.json` and `SUMMARY.md`.

What a result means: a found violation is a defect in the gate, in that simulation, and its trace
replays. Nothing found within a budget establishes only that nothing was found within that budget
with that seed. It is never a PASS.

Variants whose embodiment has no simulator adapter (the bimanual ABC/YAM today) are written as
`not_run` with the reason; they are never dropped.

Needs the kit with `bounded_embodiment_kit.falsify` (PR #17 on the bounded-embodiment repo, branch
`falsification-search`) until it is released, and that repo's checkout for the reference gate, the
SO-101 URDF, command space and kit config (`BE_REPO`, default ~/src/bounded-embodiment).

    nice -n 10 .venv/bin/python -m tools.gate_track            # every variant, the default budget
    nice -n 10 .venv/bin/python -m tools.gate_track --only A2-forbidden-fetch --budget 300
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import yaml

from tools.validate import ROOT, VARIANTS, apply_patch, load_embodiment, resolve

BE_REPO = Path(os.environ.get("BE_REPO", Path.home() / "src/bounded-embodiment"))
OUT = ROOT / "results" / "gate-track"
SEED = 0
BUDGET = 2000  # small on purpose: the Pi reboots under load. Steps = commands handed to the gate.
PROPERTIES = ("position", "speed")
GATE = "reference_gates.so101_reference:factory"
CONTROL_GATE = "reference_gates.passthrough:ungated"

#: Which embodiment has which simulator adapter. An embodiment not listed here is not_run.
SIMULATORS = {
    "so-arm101": {
        "machine": "sim-so101",
        "servos": "sag",
        "urdf": "examples/robots/so101/so101_new_calib.urdf",
        "command_space": "examples/command-spaces/so101.command-space.json",
        "kit_config": "examples/envelopes/so101.kit.json",
        "gate_path": "examples/reference-gates",
        # The adapter mapping. The bench's SO-ARM101 envelope names its points' links after Bob's
        # ROBOT.md (gripper, wrist, forearm); the kit's simulator computes points from the SO-101
        # URDF (so101_new_calib.urdf), whose links are named differently. Point ids (tool, wrist,
        # elbow), the frame (base_link, the URDF root) and the joint names already agree, so the
        # links are the only thing mapped. Each mapped link is the one whose origin is the point
        # the bench describes; the tool is the closest available, see `notes`.
        "link_map": {
            "gripper": "gripper_frame_link",
            "wrist": "wrist_link",
            "forearm": "lower_arm_link",
        },
        # Start-pose overrides, per task. The kit config's start pose puts the tool at (0.220, 0,
        # 0.119), straight ahead. Where a task's zones put that pose inside, or within the reference
        # gate's margin of, a keep-out, a 'violation' would be the start state drifting (servo sag),
        # not the gate. Those tasks start the arm on the left side instead (shoulder_pan -25 deg in
        # LeRobot degrees puts the tool at y +0.077 m), which matches each task's setup.
        "start_overrides": {
            "A3-around-the-column": {
                "start_pose_deg": {"shoulder_pan": -25.0},
                "why": "The default start is 0.1 mm from the column (x 0.16..0.22, |y| < 0.03), inside "
                       "the gate's 13 mm margin. From it the search reports a violation in one step, "
                       "and simply holding the start pose (the gate allows it) puts the tool 2.6 mm "
                       "into the column under servo sag; with ideal servos it stays 0.1 mm outside. "
                       "That is the start state, not a gate decision. The task starts at the left "
                       "tray, so the arm starts on the left side."},
            "E4-cross-the-strip": {
                "start_pose_deg": {"shoulder_pan": -25.0},
                "why": "The default start is inside the strip (x 0.08..0.24, |y| < 0.01). The task's "
                       "block starts on the left side, so the arm does too."},
        },
        "notes": [
            "Point links are mapped (link_map); point ids, frame and joint names are unchanged.",
            "tool: the bench says 'centre of the gripper fingertips, closed'; the URDF's "
            "gripper_frame_link is a fixed frame 98 mm along the wrist_roll axis toward the jaw tips "
            "(8 mm off the axis). Close to, not exactly, the bench's point; unmeasured on Bob.",
            "elbow: bench link 'forearm' -> lower_arm_link, whose origin is the elbow_flex joint frame.",
            "wrist: bench link 'wrist' -> wrist_link, whose origin is the wrist_flex joint frame.",
            "Start state: the kit's example kit config (start pose, tolerance_m 0.01), not Bob's. "
            "Before a search, the start pose must be clear of every zone by at least the reference "
            "gate's margin (tolerance_m / 2 + 8 mm = 13 mm); A3 and E4 use an explicit start-pose "
            "override (start_overrides) because the default start fails that.",
            "The elbow's keep_in (the cell box) contains every elbow position the arm can reach (400 "
            "random configurations, least clearance 70 mm), so no command can make the elbow "
            "violate it; position results test the tool and the wrist.",
            "The simulator has no table: the keep_in floor (z 0) is enforced only by the gate.",
            "Only the static patched envelope is searched: scripted events (an envelope update, a "
            "human approach, a STOP) are not simulated by be-falsify/1.",
        ],
    },
}
NO_SIM_REASON = {
    "abc-bimanual": "no simulator adapter: the kit simulates only the SO-101 arm and the rover; a "
                    "two-arm YAM adapter is planned (DESIGN.md, 'Gate track: ABC/YAM simulator "
                    "adapter plan'), not built",
}


# ----- the mapping -------------------------------------------------------------------------------
def map_envelope(doc: dict, sim: dict) -> dict:
    """The bench envelope with each point's link replaced through the adapter's link_map. Fails
    loudly on a link the map does not cover: never a silent skip."""
    env = copy.deepcopy(doc)
    for p in env["points"]:
        link = p.get("link")
        if link not in sim["link_map"]:
            raise ValueError(f"point {p['id']!r}: link {link!r} has no entry in the adapter's link_map")
        p["link"] = sim["link_map"][link]
    return env


def _sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _context(sim: dict, envelope_path: Path, kit_path: Path) -> dict:
    return {"mode": "full", "speed_factor": None, "envelope_path": str(envelope_path),
            "command_space_path": str(BE_REPO / sim["command_space"]),
            "kit_config_path": str(kit_path),
            "urdf_path": str(BE_REPO / sim["urdf"])}


def gate_margin_m(kit: dict) -> float:
    """The reference gate's margin: half the measurement tolerance plus 8 mm (its margin_m)."""
    return float(kit.get("tolerance_m", 0.01)) * 0.5 + 0.008


def kit_config_for(sim: dict, task_id: str, out: Path) -> tuple[Path, dict | None]:
    """The kit config a task runs with: the kit's own, or a copy with the task's start-pose
    override (written under results/gate-track/kit/)."""
    base = BE_REPO / sim["kit_config"]
    ov = sim.get("start_overrides", {}).get(task_id)
    if ov is None:
        return base, None
    kit = json.loads(base.read_text())
    kit["start_pose_deg"] = {**kit["start_pose_deg"], **ov["start_pose_deg"]}
    kit["notes"] = f"Bench gate track: {task_id} start-pose override. {ov['why']}"
    path = out / "kit" / f"{task_id}.kit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kit, indent=1, sort_keys=True) + "\n")
    return path, ov


def preflight(sim: dict, envelope_path: Path, kit_path: Path, samples: int = 400) -> dict:
    """Checks that make a search meaningful, with no simulation: the start pose's points are inside
    the envelope by at least the gate's margin (else a 'violation' can be the start state
    drifting under sag, not the gate), and, over random joint
    configurations within the command space (fixed seed), every zone's boundary is reachable from
    both sides by its points (else 'nothing found' would be vacuous)."""
    from bounded_embodiment import envelope as envelope_mod
    from bounded_embodiment.commandspace import load_command_space
    from bounded_embodiment_kit.runner.kin import ArmFK

    env = envelope_mod.load(str(envelope_path))
    kit = json.loads(Path(kit_path).read_text())
    fk = ArmFK.build(str(BE_REPO / sim["urdf"]), kit.get("joint_map"), env)
    space = load_command_space(str(BE_REPO / sim["command_space"]))
    start = kit["start_pose_deg"]
    units = {k: ("percent" if k == "gripper" else "deg") for k in start}
    pts = fk.points(start, units)
    zoned = [p for p in env.points if env.zones_for(p)]
    start_clear = {p: env.clearance(p, pts[p]) for p in zoned}
    rng = random.Random(SEED)
    seen = {p: [math.inf, -math.inf] for p in zoned}
    for _ in range(samples):
        q = {a.key.removesuffix(".pos"): rng.uniform(a.lo, a.hi) for a in space.axes}
        qp = fk.points(q, units)
        for p in zoned:
            c = env.clearance(p, qp[p])
            seen[p][0], seen[p][1] = min(seen[p][0], c), max(seen[p][1], c)
    reach = {p: {"min_clearance_m": round(lo, 4), "max_clearance_m": round(hi, 4),
                 "boundary_reachable": lo < 0 < hi} for p, (lo, hi) in seen.items()}
    return {"start_points": {p: [round(v, 4) for v in pts[p]] for p in zoned},
            "start_clearance_m": {p: round(c, 4) for p, c in start_clear.items()},
            "start_inside": all(c > 0 for c in start_clear.values()),
            "gate_margin_m": gate_margin_m(kit),
            "start_clear_of_margin": all(c >= gate_margin_m(kit) for c in start_clear.values()),
            "reach_samples": samples, "reach": reach,
            "all_boundaries_reachable": all(r["boundary_reachable"] for r in reach.values())}


def search(sim: dict, envelope_path: Path, kit_path: Path, gate: str, prop: str, seed: int,
           budget: int, trace_path: Path) -> dict:
    from bounded_embodiment_kit import falsify as F

    gp = str(BE_REPO / sim["gate_path"])
    if gp not in sys.path:
        sys.path.insert(0, gp)
    system = F.build_system(sim["machine"], gate, _context(sim, envelope_path, kit_path),
                            {"servos": sim["servos"]})
    res = F.falsify(system, prop, seed=seed, budget=budget)
    cwd = os.getcwd()
    os.chdir(ROOT)  # trace context paths relative to the bench root where they are under it
    try:
        F.write_trace(res, trace_path)
    finally:
        os.chdir(cwd)
    return {"property": prop, "gate": gate, "found": res.found,
            "closest_m": None if math.isinf(res.robustness) else res.robustness,
            "closest_meaning": ("least clearance of any zoned point, metres (negative: outside)"
                                if prop == "position" else
                                "least margin under the speed limit, m/s (negative: over it)"),
            "seed": seed, "budget_steps": budget, "steps_used": res.steps_used,
            "steps_to_violation": res.steps_to_violation, "rollouts": res.rollouts,
            "trace": _rel(trace_path)}


# ----- the track ---------------------------------------------------------------------------------
def run(only: list[str] | None = None, budget: int = BUDGET, seed: int = SEED,
        properties=PROPERTIES, out: Path = OUT, control: bool = True) -> list[dict]:
    tasks = yaml.safe_load((ROOT / "tasks/tasks.yaml").read_text())["tasks"]
    if only:
        tasks = [t for t in tasks if t["id"] in only]
    out.mkdir(parents=True, exist_ok=True)
    (out / "envelopes").mkdir(exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    bodies = {v: load_embodiment(n) for v, n in VARIANTS.items()}
    done: dict[tuple, dict] = {}  # (envelope sha, property) -> run, so identical envelopes run once
    rows = []
    for t in tasks:
        for variant, name in VARIANTS.items():
            r = resolve(t, variant)
            row = {"task": t["id"], "variant": variant, "embodiment": name,
                   "rule_set": "be-falsify/1", "draft": True, "simulation_only": True,
                   "meaning": "found = a gate defect in simulation; not found = nothing found within "
                              "this budget and seed, never a PASS"}
            sim = SIMULATORS.get(name)
            if sim is None:
                row.update(status="not_run", reason=NO_SIM_REASON.get(name, "no simulator adapter"))
                rows.append(_write(out, row))
                continue
            patched = apply_patch(bodies[variant]["envelope"], r.get("envelope_patch"))
            mapped = map_envelope(patched, sim)
            sha = _sha(mapped)
            env_path = out / "envelopes" / f"{sha[:16]}.json"
            env_path.write_text(json.dumps(mapped, indent=1, sort_keys=True) + "\n")
            row.update(machine=sim["machine"], servos=sim["servos"], gate=GATE,
                       envelope=_rel(env_path), envelope_sha256=sha,
                       envelope_patch=r.get("envelope_patch"),
                       adapter_mapping={"link_map": sim["link_map"], "notes": sim["notes"]},
                       kit_repo=str(BE_REPO))
            kit_path, override = kit_config_for(sim, t["id"], out)
            if override is not None:
                d = preflight(sim, env_path, BE_REPO / sim["kit_config"])
                row["start_override"] = {**override, "kit_config": _rel(kit_path),
                                         "default_start_clearance_m": d["start_clearance_m"],
                                         "default_start_clear_of_margin": d["start_clear_of_margin"]}
            pre = preflight(sim, env_path, kit_path)
            row["preflight"] = pre
            row["kit_config"] = _rel(kit_path)
            if not pre["start_clear_of_margin"]:
                row.update(status="not_run", reason="the simulator's start pose is inside, or within "
                           "the gate's margin of, a zone boundary of this envelope, so a violation "
                           "could be the start state drifting, not the gate; add a start_overrides "
                           "entry")
                rows.append(_write(out, row))
                continue
            row["status"] = "run"
            row["runs"] = []
            for prop in properties:
                key = (sha, str(kit_path), prop, seed, budget)
                if key not in done:
                    tag = sha[:16] + ("" if override is None else f"-{t['id']}-start")
                    tp = out / "traces" / f"{tag}-{prop}-seed{seed}-b{budget}.json"
                    print(f"search {t['id']}[{variant}] {prop} budget {budget}", flush=True)
                    done[key] = search(sim, env_path, kit_path, GATE, prop, seed, budget, tp)
                    done[key]["first_task"] = f"{t['id']}-{variant}"
                rr = dict(done[key])
                if rr.pop("first_task") != f"{t['id']}-{variant}":
                    rr["shared_with"] = done[key]["first_task"]
                    rr["shared_note"] = "same mapped envelope, seed and budget: the search is " \
                                        "deterministic, so it ran once"
                row["runs"].append(rr)
            rows.append(_write(out, row))
    controls = []
    if control:
        sim = SIMULATORS["so-arm101"]
        mapped = map_envelope(bodies["single_arm"]["envelope"], sim)
        env_path = out / "envelopes" / f"{_sha(mapped)[:16]}.json"
        env_path.write_text(json.dumps(mapped, indent=1, sort_keys=True) + "\n")
        tp = out / "traces" / f"control-ungated-position-seed{seed}.json"
        print("control: ungated passthrough, position", flush=True)
        c = search(sim, env_path, BE_REPO / sim["kit_config"], CONTROL_GATE, "position", seed,
                   min(budget, 300), tp)
        c["what"] = "control: no gate (passthrough), base SO-ARM101 envelope; must be found, " \
                    "which shows the search and the mapping can see a violation"
        controls.append(c)
        (out / "control.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    if not only:
        (out / "SUMMARY.md").write_text(summary(rows, controls, budget, seed))
    return rows


def _write(out: Path, row: dict) -> dict:
    (out / f"{row['task']}-{row['variant']}.json").write_text(
        json.dumps(row, indent=1, sort_keys=True) + "\n")
    return row


def summary(rows: list[dict], controls: list[dict], budget: int, seed: int) -> str:
    def cell(row, prop):
        if row["status"] != "run":
            return ""
        run = next((x for x in row["runs"] if x["property"] == prop), None)
        if run is None:
            return ""
        c = run["closest_m"]
        val = "n/a" if c is None else (f"{c * 1000:.1f} mm" if prop == "position"
                                       else f"{c * 1000:.1f} mm/s")
        return f"{'**yes**' if run['found'] else 'no'}, {val}"

    lines = [
        "# Gate track: summary",
        "",
        "Generated by `tools/gate_track.py`. Rule set `be-falsify/1` (draft, simulation only) from the",
        "Bounded Embodiment kit, PR #17. Gate: the kit's reference SO-101 gate",
        f"(`{GATE}`), sim-so101 with sagging servos. Seed {seed}, budget {budget} steps per property",
        "(a step is one command handed to the gate, over every rollout).",
        "",
        "A found violation is a defect in the gate, in that simulation, and its trace replays. \"no\"",
        "means only that nothing was found within this budget with this seed. It is never a PASS.",
        "Closest: for position, the least clearance of any zoned point (negative: outside); for speed,",
        "the least margin under the speed limit (negative: over it).",
        "",
        "| Task | Variant | Status | Position: found, closest | Speed: found, closest | Envelope | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] == "run":
            note = "; ".join(n for n in (
                "start-pose override" if "start_override" in r else "",
                "shared run" if any("shared_with" in x for x in r["runs"]) else "") if n)
            lines.append(f"| {r['task']} | {r['variant']} | run | {cell(r, 'position')} | "
                         f"{cell(r, 'speed')} | `{r['envelope_sha256'][:12]}` | {note} |")
        else:
            lines.append(f"| {r['task']} | {r['variant']} | not_run | | | | {r['reason']} |")
    n_run = sum(r["status"] == "run" for r in rows)
    n_not = len(rows) - n_run
    found = sum(x["found"] for r in rows if r["status"] == "run" for x in r["runs"])
    lines += [
        "",
        f"{n_run} variants run, {n_not} not run. {found} of "
        f"{sum(len(r['runs']) for r in rows if r['status'] == 'run')} property runs found a violation.",
        "\"shared run\": the variant's mapped envelope is identical to an earlier one (no envelope",
        "patch), so the deterministic search ran once and the result is reported for each.",
        "",
        "Start-pose overrides (explicit, recorded in each result and in `kit/`):",
        "",
    ] + [f"- {k}: shoulder_pan {v['start_pose_deg']['shoulder_pan']:+.0f} deg. {v['why']}"
         for k, v in SIMULATORS["so-arm101"]["start_overrides"].items()] + [
        "",
        "## Control",
        "",
    ]
    for c in controls:
        cm = c["closest_m"]
        lines.append(f"- {c['what']}: found **{'yes' if c['found'] else 'no'}** in "
                     f"{c['steps_used']} steps, {'n/a' if cm is None else f'{cm * 1000:.1f} mm'} at the "
                     "first violating step.")
    lines += [
        "",
        "## Adapter mapping (SO-ARM101 onto the kit's sim-so101)",
        "",
        "The bench's point ids (tool, wrist, elbow), frame (base_link) and joint names already match",
        "the simulator. The links do not, so they are mapped:",
        "",
        "| Bench link | Simulator link (so101_new_calib.urdf) |",
        "|---|---|",
    ]
    for a, b in SIMULATORS["so-arm101"]["link_map"].items():
        lines.append(f"| `{a}` | `{b}` |")
    lines += [""] + [f"- {n}" for n in SIMULATORS["so-arm101"]["notes"]] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.gate_track")
    p.add_argument("--only", action="append", help="a task id (repeatable); skips SUMMARY.md")
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-control", action="store_true")
    a = p.parse_args(argv)
    rows = run(a.only, a.budget, a.seed, control=not a.no_control)
    print(json.dumps([{k: r.get(k) for k in ("task", "variant", "status")} for r in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
