"""Run the pack-with-hand task in Isaac Lab-Arena, score every episode with the same oracle, and
compare each verdict with the committed MuJoCo one. Needs Isaac Lab-Arena on a GPU box (``BREV.md``).

    python -m sim.yam_pack_arena.isaac.run --check-fk                    # USD in Isaac vs MuJoCo FK
    python -m sim.yam_pack_arena.isaac.run --seeds 3                     # 4 policies x 3 seeds
    python -m sim.yam_pack_arena.isaac.run --seeds 20 --out results/sim-track/yam-pack-isaac

Arena's own flags pass through (``--device cpu`` is the default here: PhysX on the CPU is
repeatable run to run, PhysX on the GPU is not bitwise, and one environment gains nothing on a GPU).
Writes ``<out>/<policy>.json`` (one record per seed, MuJoCo's format plus ``mujoco_verdict``) and
``<out>/SUMMARY.md``; exits 1 if any verdict differs from MuJoCo's or ``--check-fk`` fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENV_CLASS = "sim.yam_pack_arena.isaac.environment:YamPackEnvironment"
POLICIES = ["comply", "ignore", "right_only", "do_nothing"]


def _own_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--out", type=Path, default=ROOT / "results/sim-track/yam-pack-isaac")
    ap.add_argument("--check-fk", action="store_true")
    ap.add_argument("--fk-samples", type=int, default=50)
    ours, rest = ap.parse_known_args(argv)
    if "--device" not in rest:
        rest += ["--device", "cpu"]
    if "--num_envs" not in rest:
        rest += ["--num_envs", "1"]
    if "--external_environment_class_path" not in rest:
        rest += ["--external_environment_class_path", ENV_CLASS, "yam_pack"]
    return ours, rest


def make_env(rest):
    """Build the env the way Arena's policy_runner does. Returns (env, simulation app context)."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    sys.argv = [sys.argv[0], *rest]
    parser = get_isaaclab_arena_cli_parser()
    args, _ = parser.parse_known_args()
    ctx = SimulationAppContext(args)
    ctx.__enter__()
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, _ = parser.parse_known_args()
    env = get_arena_builder_from_cli(args).make_registered()
    u = env.unwrapped
    assert not getattr(u, "_physics_handles_decimation", False), "the driver needs one apply_actions per substep"
    return env, ctx


def check_fk(env, n: int) -> dict:
    """Random joint values written into Isaac's left arm; named points read back from PhysX and
    compared with MuJoCo's FK of the same MJCF placed at the same base. This checks Isaac's reading
    of ``yam.usda`` (joint frames, axes, signs, fixed base), which ``usd_fk.py`` cannot."""
    import mujoco
    import numpy as np
    import torch

    from sim.yam_pack.scene import build_model

    u = env.unwrapped
    term = u.action_manager.get_term("driver")
    be = term.backend
    m = build_model(term.spec, [b[:2] for b in term.spec["objects"]["blocks"]["at"]])
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        q = {}
        for s in ("left", "right"):
            ids = [m.joint(f"{s}_joint{j}").id for j in range(1, 7)]
            q[s] = np.array([rng.uniform(*m.jnt_range[i]) for i in ids])
            for i, v in zip(ids, q[s]):
                d.qpos[m.jnt_qposadr[i]] = v
            t = torch.tensor(q[s], dtype=torch.float32, device=u.device).reshape(1, -1)
            be.arm[s].write_joint_position_to_sim_index(position=t, joint_ids=be.jids[s], env_ids=be.env_ids)
        u.sim.forward()
        u.scene.update(dt=u.physics_dt)
        mujoco.mj_kinematics(m, d)
        for s in ("left", "right"):
            mj = {"tool": d.site_xpos[m.site(f"{s}_grasp_site").id], "wrist": d.xpos[m.body(f"{s}_link_5").id],
                  "elbow": d.xpos[m.body(f"{s}_link_3").id]}
            for p, x in mj.items():
                worst = max(worst, float(np.linalg.norm(be.point(f"{s}_{p}") - x)))
    return {"samples": n, "worst_point_gap_m": worst, "ok": worst < 1e-4}


def episode(env, policy, seed: int, spec: dict) -> tuple[list[dict], dict, float]:
    u = env.unwrapped
    term = u.action_manager.get_term("driver")
    term.seed, term.overrides = seed, None
    env.reset()
    policy.reset()
    t0 = time.time()
    settle, end_at = 1.0, None
    while True:
        term.sync()
        drv = term.ensure_driver()
        if drv.t >= spec["episode"]["max_time"]:
            break
        env.step(policy.get_action(env, None))
        term.sync()
        if end_at is None and policy.done() and drv.hand_done:
            end_at = drv.t + settle
        if end_at is not None and drv.t >= end_at:
            break
    drv.flush()
    info = {"seed": seed, "first_release_t": drv.first_release_t, "hand_target": drv.hand_target.tolist(),
            "hand_arrive_s": drv.hand_arrive, "block_xy": drv.block_xy}
    return drv.trace, info, time.time() - t0


def main(argv=None) -> int:
    ours, rest = _own_args(list(sys.argv[1:] if argv is None else argv))
    env, ctx = make_env(rest)
    try:
        if ours.check_fk:
            r = check_fk(env, ours.fk_samples)
            print(json.dumps(r))
            return 0 if r["ok"] else 1

        from sim.yam_pack.oracle import score, sha256_json
        from sim.yam_pack.run import ENVELOPE, OUT as MUJOCO_OUT
        from sim.yam_pack.spec import load_spec
        from sim.yam_pack_arena.isaac.policy import ReferencePolicy, ReferencePolicyCfg

        spec = load_spec()
        envelope = json.loads(ENVELOPE.read_text())
        ours.out.mkdir(parents=True, exist_ok=True)
        mismatches, lines = 0, []
        for name in ours.policies:
            mj = {r["seed"]: r for r in json.loads((MUJOCO_OUT / f"{name}.json").read_text())}
            pol = ReferencePolicy(ReferencePolicyCfg(mode=name))
            recs = []
            for seed in range(ours.seeds):
                trace, info, wall = episode(env, pol, seed, spec)
                v = score(trace, envelope, spec)
                rec = {**info, "policy": name, "verdict": v["verdict"],
                       "checks": {k: c["result"] for k, c in v["checks"].items()},
                       "why": {k: c["why"] for k, c in v["checks"].items()},
                       "violations": v["violations"], "metrics": v["metrics"],
                       "trace_sha256": sha256_json(trace), "wall_s": round(wall, 1),
                       "mujoco_verdict": mj.get(seed, {}).get("verdict"),
                       "mujoco_checks": mj.get(seed, {}).get("checks")}
                same = rec["verdict"] == rec["mujoco_verdict"]
                mismatches += 0 if same else 1
                print(f"{name} seed {seed}: Isaac {rec['verdict']}, MuJoCo {rec['mujoco_verdict']}"
                      f"{'' if same else '  <-- differs'} ({wall:.0f} s)", flush=True)
                recs.append(rec)
            (ours.out / f"{name}.json").write_text(json.dumps(recs, indent=1))
            agree = sum(r["verdict"] == r["mujoco_verdict"] for r in recs)
            checks_agree = sum(r["checks"] == r["mujoco_checks"] for r in recs)
            lines.append(f"| {name} | {sum(r['verdict'] == 'PASS' for r in recs)} | "
                         f"{sum(r['verdict'] == 'FAIL' for r in recs)} | {agree}/{len(recs)} | {checks_agree}/{len(recs)} |")
        (ours.out / "SUMMARY.md").write_text("\n".join([
            "# Sim track: pack with a hand mid-task in Isaac Lab-Arena",
            "",
            "Same spec, seeds, policies, envelope and oracle as results/sim-track/yam-pack (MuJoCo);",
            "grasp is a kinematic carry instead of a weld (sim/yam_pack_arena/core.py).",
            "",
            "| Policy | PASS | FAIL | Verdict same as MuJoCo | All checks same |",
            "|---|---|---|---|---|",
            *lines, ""]))
        print(f"{mismatches} verdict(s) differ from MuJoCo")
        return 1 if mismatches else 0
    finally:
        env.close()
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
