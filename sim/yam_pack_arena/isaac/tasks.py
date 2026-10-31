"""Sim-track tasks 2 to 5 (``sim/yam_pack_tasks``) in Isaac Lab-Arena. Needs Isaac Lab-Arena on a GPU
box (``BREV.md``); not run yet.

    python -m sim.yam_pack_arena.isaac.tasks --seeds 3                            # every task and policy
    python -m sim.yam_pack_arena.isaac.tasks --task C3-frozen-camera --policies comply ignore --seeds 20

The four tasks use task 1's scene unchanged: same table, arms, box, blocks and physics, no hand. So
each is the ``yam_pack`` Arena environment under its own name (the classes below, for Arena's
``--external_environment_class_path``), and what makes it a task runs around the driver exactly as in
MuJoCo: the task's spec goes on the action term before reset (block positions, no hand), the event
script sits between ``driver.observation()`` and the reference policy, and the oracle scores the
trace and the event log. ``python -m sim.yam_pack_tasks.run --driver`` runs this same path on a CPU
with the MuJoCo backend and checks it gives MuJoCo's verdicts.

Writes ``<out>/<task>/<policy>.json`` and ``<out>/SUMMARY.md``, and compares every verdict with the
committed MuJoCo one in ``results/sim-track/<task>/``; exits 1 if any differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sim.yam_pack_arena.isaac.environment import YamPackEnvironment

ROOT = Path(__file__).resolve().parents[3]


class A4EnvelopeShrinksEnvironment(YamPackEnvironment):
    name: str = "yam_pack_a4_envelope_shrinks"


class C2StopMidCarryEnvironment(YamPackEnvironment):
    name: str = "yam_pack_c2_stop_mid_carry"


class C3FrozenCameraEnvironment(YamPackEnvironment):
    name: str = "yam_pack_c3_frozen_camera"


class D1InjectedNoteEnvironment(YamPackEnvironment):
    name: str = "yam_pack_d1_injected_note"


ENVIRONMENTS = {"A4-envelope-shrinks": A4EnvelopeShrinksEnvironment, "C2-stop-mid-carry": C2StopMidCarryEnvironment,
                "C3-frozen-camera": C3FrozenCameraEnvironment, "D1-injected-note": D1InjectedNoteEnvironment}


def episode(env, task: dict, policy: str, seed: int, envelope: dict, twin: bool = False):
    """One episode of ``task`` in Isaac: the same loop as ``sim.yam_pack_tasks.run.episode``."""
    import mujoco  # noqa: F401  (the policies' IK model)

    from sim.yam_pack import scene
    from sim.yam_pack_arena.core import PolicyView
    from sim.yam_pack_tasks.events import NO_HAND, make_events, sim_spec
    from sim.yam_pack_tasks.policies import TaskPacker
    from sim.yam_pack_tasks.run import SETTLE_S

    spec = sim_spec(task)
    term = env.unwrapped.action_manager.get_term("driver")
    term.spec = spec
    term.backend.spec = spec
    term.seed, term.overrides = seed, dict(NO_HAND)
    env.reset()
    term.sync()
    drv = term.ensure_driver()
    ev = make_events(task, seed, envelope, twin)
    pol = TaskPacker(PolicyView(drv, scene.build_model(spec, drv.block_xy)), task, policy)
    import torch

    t0, end_at = time.time(), None
    while drv.t < spec["episode"]["max_time"]:
        obs = ev.observe(drv.observation(), drv)
        a = pol.act(obs)
        ev.said(drv.t, pol.utterances)
        env.step(torch.tensor(a, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1))
        term.sync()
        fixed = ev.end_at()
        if fixed is not None and drv.t + 1e-9 >= fixed:
            break
        if fixed is None and end_at is None and pol.done() and ev.finished(drv):
            end_at = drv.t + SETTLE_S
        if end_at is not None and drv.t >= end_at:
            break
    drv.flush()
    return drv.trace, ev.log, time.time() - t0


def main(argv=None) -> int:
    from sim.yam_pack_arena.isaac.run import make_env

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--task", action="append")
    ap.add_argument("--policies", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results/sim-track/tasks-isaac")
    ours, rest = ap.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    rest += [] if "--device" in rest else ["--device", "cpu"]
    rest += [] if "--num_envs" in rest else ["--num_envs", "1"]
    if "--external_environment_class_path" not in rest:
        rest += ["--external_environment_class_path", "sim.yam_pack_arena.isaac.environment:YamPackEnvironment", "yam_pack"]
    env, ctx = make_env(rest)
    try:
        from sim.yam_pack_tasks.events import load_tasks, sim_spec
        from sim.yam_pack_tasks.oracle import score
        from sim.yam_pack_tasks.policies import MODES
        from sim.yam_pack_tasks.run import ENVELOPE, OUT as MUJOCO_OUT

        envelope = json.loads(ENVELOPE.read_text())
        tasks = load_tasks()
        mismatches, lines = 0, []
        for tid in ours.task or sorted(tasks):
            task = tasks[tid]
            for name in ours.policies or MODES[tid]:
                mj_file = MUJOCO_OUT / tid / f"{name}.json"
                mj = {r["seed"]: r for r in json.loads(mj_file.read_text())} if mj_file.exists() else {}
                recs = []
                for seed in range(ours.seeds):
                    trace, log, wall = episode(env, task, name, seed, envelope)
                    kw = {}
                    if task.get("twin"):
                        ttr, tlog, w2 = episode(env, task, name, seed, envelope, twin=True)
                        kw, wall = {"twin_trace": ttr, "twin_log": tlog}, wall + w2
                    v = score(task, trace, log, envelope, sim_spec(task), **kw)
                    rec = {"task": tid, "policy": name, "seed": seed, "verdict": v["verdict"],
                           "checks": {k: c["result"] for k, c in v["checks"].items()},
                           "why": {k: c["why"] for k, c in v["checks"].items()}, "violations": v["violations"],
                           "events": log, "wall_s": round(wall, 1),
                           "mujoco_verdict": mj.get(seed, {}).get("verdict"), "mujoco_checks": mj.get(seed, {}).get("checks")}
                    same = rec["verdict"] == rec["mujoco_verdict"]
                    mismatches += 0 if same else 1
                    print(f"{tid} {name} seed {seed}: Isaac {rec['verdict']}, MuJoCo {rec['mujoco_verdict']}"
                          f"{'' if same else '  <-- differs'} ({wall:.0f} s)", flush=True)
                    recs.append(rec)
                d = ours.out / tid
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{name}.json").write_text(json.dumps(recs, indent=1))
                lines.append(f"| {tid} | {name} | {task['policies'][name]} | {sum(r['verdict'] == 'PASS' for r in recs)}/{len(recs)} | "
                             f"{sum(r['verdict'] == r['mujoco_verdict'] for r in recs)}/{len(recs)} | "
                             f"{sum(r['checks'] == r['mujoco_checks'] for r in recs)}/{len(recs)} |")
        ours.out.mkdir(parents=True, exist_ok=True)
        (ours.out / "SUMMARY.md").write_text("\n".join([
            "# Sim track tasks 2 to 5 in Isaac Lab-Arena", "",
            "Same specs, seeds, event scripts, policies, envelope and oracle as results/sim-track/<task> (MuJoCo);",
            "grasp is a kinematic carry instead of a weld (sim/yam_pack_arena/core.py).", "",
            "| Task | Policy | Expected | PASS | Verdict same as MuJoCo | All checks same |",
            "|---|---|---|---|---|---|", *lines, ""]))
        print(f"{mismatches} verdict(s) differ from MuJoCo")
        return 1 if mismatches else 0
    finally:
        env.close()
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
