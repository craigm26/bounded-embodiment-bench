"""Run tasks 2 to 5 in MuJoCo and score every episode.

    python -m sim.yam_pack_tasks.run                                  # 4 tasks x their policies x 20 seeds, twice each
    python -m sim.yam_pack_tasks.run --task C3-frozen-camera --policy comply --seeds 1 --no-write
    python -m sim.yam_pack_tasks.run --driver                         # also through the Arena driver (CPU), verdicts must match

Each episode runs twice from its seed; the traces, event logs and verdicts must hash the same.
``--driver`` runs every episode a third time through ``sim/yam_pack_arena``'s ``PackDriver`` with the
MuJoCo backend (soft-weld carry instead of welds, the path Isaac Lab-Arena takes) and checks it gets
the same verdict and the same result on every check. Results go to ``results/sim-track/<task id>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco

from sim.yam_pack.oracle import sha256_json
from sim.yam_pack.scene import World
from sim.yam_pack_tasks.events import NO_HAND, load_tasks, make_events, sim_spec
from sim.yam_pack_tasks.oracle import score
from sim.yam_pack_tasks.policies import MODES, TaskPacker

ROOT = Path(__file__).resolve().parents[2]
ENVELOPE = ROOT / "embodiments/abc-bimanual/envelope.json"
OUT = ROOT / "results/sim-track"
SETTLE_S = 1.0


def episode(task: dict, policy: str, seed: int, twin: bool = False, driver: bool = False,
            envelope: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    envelope = envelope or json.loads(ENVELOPE.read_text())
    spec = sim_spec(task)
    ev = make_events(task, seed, envelope, twin)
    if driver:
        from sim.yam_pack.episode import draw
        from sim.yam_pack_arena.core import MujocoBackend, PackDriver, PolicyView

        dr = draw(spec, seed, NO_HAND)
        be = MujocoBackend(spec, dr.block_xy)
        ep = PackDriver(be, spec, seed, draws=dr)
        pol = TaskPacker(PolicyView(ep, be.m), task, policy)
        step = lambda a: ep.step(a, be.physics_step)  # noqa: E731
    else:
        ep = World(spec, seed, NO_HAND)
        pol = TaskPacker(ep, task, policy)
        step = ep.step
    end_at = None
    while ep.t < spec["episode"]["max_time"]:
        obs = ev.observe(ep.observation(), ep)
        a = pol.act(obs)
        ev.said(ep.t, pol.utterances)
        step(a)
        fixed = ev.end_at()
        if fixed is not None and ep.t + 1e-9 >= fixed:
            break
        if fixed is None and end_at is None and pol.done() and ev.finished(ep):
            end_at = ep.t + SETTLE_S
        if end_at is not None and ep.t >= end_at:
            break
    if driver:
        ep.flush()
    info = {"seed": seed, "policy": policy, "twin": twin, "first_release_t": ep.first_release_t,
            "block_xy": ep.block_xy}
    return ep.trace, ev.log, info


def run_one(task: dict, policy: str, seed: int, envelope: dict, driver: bool = False) -> dict:
    spec = sim_spec(task)
    twin = bool(task.get("twin"))

    def once(drv=False):
        tr, log, info = episode(task, policy, seed, driver=drv, envelope=envelope)
        kw = {}
        if twin:
            ttr, tlog, _ = episode(task, policy, seed, twin=True, driver=drv, envelope=envelope)
            kw = {"twin_trace": ttr, "twin_log": tlog}
            info["twin_trace_sha256"] = sha256_json(ttr)
        v = score(task, tr, log, envelope, spec, **kw)
        return tr, log, info, v

    tr1, log1, info, v1 = once()
    tr2, log2, info2, v2 = once()
    det = (sha256_json(tr1) == sha256_json(tr2) and sha256_json(log1) == sha256_json(log2)
           and v1["verdict_sha256"] == v2["verdict_sha256"]
           and info.get("twin_trace_sha256") == info2.get("twin_trace_sha256"))
    row = {**info, "task": task["id"], "verdict": v1["verdict"],
           "checks": {k: c["result"] for k, c in v1["checks"].items()},
           "why": {k: c["why"] for k, c in v1["checks"].items()},
           "violations": v1["violations"], "metrics": v1["metrics"], "events": log1,
           "trace_sha256": sha256_json(tr1), "deterministic": det}
    if driver:
        _, _, _, vd = once(drv=True)
        row["driver"] = {"verdict": vd["verdict"], "checks": {k: c["result"] for k, c in vd["checks"].items()},
                         "violation_kinds": sorted(vd["violations"])}
        row["driver_agrees"] = vd["verdict"] == v1["verdict"] and row["driver"]["checks"] == row["checks"]
    return row


def _job(args):
    tid, policy, seed, driver = args
    task = load_tasks()[tid]
    return run_one(task, policy, seed, json.loads(ENVELOPE.read_text()), driver)


def summary(task: dict, rows: dict, seeds: list[int], driver: bool) -> str:
    exp = task["policies"]
    lines = [
        f"# Sim track: {task['id']}, bimanual, in the pack scene (MuJoCo)",
        "",
        f"{task['title']}. MuJoCo {mujoco.__version__}, task 1's scene and YAM model (`sim/yam_pack`), no hand;",
        "the event and the checks are in `sim/yam_pack_tasks/` (README there). Envelope:",
        "embodiments/abc-bimanual/envelope.json (placeholder numbers).",
        f"Seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} per policy); every episode run twice from its seed"
        + (", plus its twin with no STOP" if task.get("twin") else "") + ".",
        "",
        "| Policy | Expected | PASS | FAIL | Deterministic |" + (" Arena driver agrees |" if driver else ""),
        "|---|---|---|---|---|" + ("---|" if driver else ""),
    ]
    for p, rs in rows.items():
        npass = sum(r["verdict"] == "PASS" for r in rs)
        line = f"| {p} | {exp[p]} | {npass} | {len(rs) - npass} | {sum(r['deterministic'] for r in rs)}/{len(rs)} |"
        if driver:
            line += f" {sum(r.get('driver_agrees', False) for r in rs)}/{len(rs)} |"
        lines.append(line)
    lines += ["", "Failing checks and violation kinds, per policy (episodes):", ""]
    for p, rs in rows.items():
        failing, kinds = {}, {}
        for r in rs:
            for k, ok in r["checks"].items():
                if not ok:
                    failing[k] = failing.get(k, 0) + 1
            for k in r["violations"]:
                kinds[k] = kinds.get(k, 0) + 1
        lines.append(f"- **{p}**: failing checks {failing or 'none'}; violations {kinds or 'none'}")
    lines += ["", "One episode per policy (seed 0), as the oracle explains it:", ""]
    for p, rs in rows.items():
        lines.append(f"- **{p}** ({rs[0]['verdict']}):")
        for k, why in rs[0]["why"].items():
            lines.append(f"  - {k} {'ok' if rs[0]['checks'][k] else 'FAIL'}: {why}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    tasks = load_tasks()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", choices=sorted(tasks))
    ap.add_argument("--policy", action="append")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--driver", action="store_true", help="also run each episode through the Arena driver (CPU)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    ok = True
    t0 = time.time()
    for tid in args.task or sorted(tasks):
        task = tasks[tid]
        pols = [p for p in MODES[tid] if not args.policy or p in args.policy]
        jobs = [(tid, p, s, args.driver) for p in pols for s in seeds]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            out = list(ex.map(_job, jobs))
        rows = {p: [r for r in out if r["policy"] == p] for p in pols}
        for p, rs in rows.items():
            bad = [r["seed"] for r in rs if r["verdict"] != task["policies"][p] or not r["deterministic"]
                   or (args.driver and not r["driver_agrees"])]
            ok &= not bad
            print(f"{tid:22s} {p:20s} {sum(r['verdict'] == 'PASS' for r in rs)}/{len(rs)} PASS, expected "
                  f"{task['policies'][p]}; unexpected, non-deterministic or driver-disagreeing seeds: {bad or 'none'}",
                  flush=True)
        if not args.no_write:
            d = OUT / tid
            d.mkdir(parents=True, exist_ok=True)
            for p, rs in rows.items():
                (d / f"{p}.json").write_text(json.dumps(rs, indent=1, sort_keys=True) + "\n")
            (d / "SUMMARY.md").write_text(summary(task, rows, seeds, args.driver))
    print(f"{time.time() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
