"""Run the pack task's reference policies on every YAM model and check the verdicts do not move.

    python -m sim.yam_pack.compare_models                  # 20 seeds x 4 policies x 3 models, ~4 min on 4 CPUs
    python -m sim.yam_pack.compare_models --seeds 2 --no-write

For each policy and seed, one episode per model, all planned by the same policies in canonical joints
and carried onto the model by its joint map (``joint_map.py``):

- ``menagerie_yam``, the canonical model. Its trace must hash the same as the committed result in
  ``results/sim-track/yam-pack/<policy>.json``: the layer changes nothing on the model it maps to itself.
- ``i2rt_yam_v1``, i2rt's own description (``models/i2rt_yam_v1.yaml``), run twice for determinism.
- ``yam_scrambled``, the canonical arm with renamed, re-zeroed and reversed joints (``models/synthetic.py``),
  built fresh for the run. Same physical arm, so its trace must stay within ``SCRAMBLED_TOL_M`` of the
  canonical one (MuJoCo's constraint solver works in the model's own joint coordinates, so joint friction
  and contacts resolve a hair differently; a wrong map moves points by centimetres).

Pass: every model gives the expected verdict on every seed and the same result on every check as the
canonical model. Also reported: the largest distance between a named point in each model's trace and
the canonical trace at the same time, and how many commands a model's joint ranges clamped.
Results go to ``results/sim-track/yam-pack/models/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from sim.yam_pack.joint_map import CANONICAL, load_config
from sim.yam_pack.models import fit, synthetic
from sim.yam_pack.oracle import score, sha256_json
from sim.yam_pack.policies import POLICIES, Packer
from sim.yam_pack.run import ENVELOPE, EXPECTED, OUT as RUN_OUT
from sim.yam_pack.scene import World
from sim.yam_pack.spec import load_spec

OUT = RUN_OUT / "models"
I2RT = "i2rt_yam_v1"
_SCRAMBLED: Path | None = None


def episode(policy: str, seed: int, spec: dict, model) -> tuple[list[dict], int]:
    """``run.episode`` on a given model; also returns how many commands the model's ranges clamped."""
    w = World(spec, seed, model=model)
    pol = Packer(w, policy)
    end_at = None
    while w.t < spec["episode"]["max_time"]:
        w.step(pol.act(w.observation()))
        if end_at is None and pol.done() and w.hand_done:
            end_at = w.t + 1.0
        if end_at is not None and w.t >= end_at:
            break
    return w.trace, w.limit_clips


SCRAMBLED_TOL_M = 1e-3


def deviation(a: list[dict], b: list[dict], lift=(0.0, 0.0, 0.0)) -> dict:
    """Largest named-point and joint difference between two traces over their common frames; the
    points also with ``a``'s model lift (``fit.check``) taken off."""
    n = min(len(a), len(b))
    d = np.array([np.subtract(a[i]["points"][p], b[i]["points"][p]) for i in range(n) for p in a[i]["points"]])
    jt = max(abs(a[i]["joints"][j] - b[i]["joints"][j]) for i in range(n) for j in a[i]["joints"])
    return {"frames": [len(a), len(b)], "point_m": float(np.linalg.norm(d, axis=1).max()),
            "point_after_lift_m": float(np.linalg.norm(d - np.asarray(lift), axis=1).max()), "joint_rad": jt}


def _summary(tr: list[dict], envelope: dict, spec: dict, clips: int) -> dict:
    v = score(tr, envelope, spec)
    return {"verdict": v["verdict"], "checks": {k: c["result"] for k, c in v["checks"].items()},
            "violations": sorted(v["violations"]), "time_to_stop_ms": v["metrics"]["time_to_stop_ms"],
            "min_human_distance_m": v["metrics"].get("min_human_distance_m"),
            "violation_events": v["metrics"]["violation_events"], "trace_sha256": sha256_json(tr),
            "limit_clips": clips}


def job(args) -> dict:
    policy, seed, scrambled = args
    spec = load_spec()
    envelope = json.loads(ENVELOPE.read_text())
    out = {"policy": policy, "seed": seed}
    canon, c_clips = episode(policy, seed, spec, CANONICAL)
    out[CANONICAL] = _summary(canon, envelope, spec, c_clips)
    t1, k1 = episode(policy, seed, spec, I2RT)
    t2, _ = episode(policy, seed, spec, I2RT)
    lift = np.array(fit.check(load_config(I2RT), n=1)["lift_mm"]) / 1000
    out[I2RT] = {**_summary(t1, envelope, spec, k1), "deterministic": sha256_json(t1) == sha256_json(t2),
                 "vs_canonical": deviation(t1, canon, lift)}
    ts, ks = episode(policy, seed, spec, load_config(scrambled))
    out["yam_scrambled"] = {**_summary(ts, envelope, spec, ks), "vs_canonical": deviation(ts, canon)}
    return out


def committed_hashes() -> dict:
    out = {}
    for p in POLICIES:
        f = RUN_OUT / f"{p}.json"
        if f.exists():
            out.update({(p, r["seed"]): r["trace_sha256"] for r in json.loads(f.read_text())})
    return out


def judge(rows: list[dict], committed: dict) -> list[str]:
    problems = []
    for r in rows:
        p, s, c = r["policy"], r["seed"], r[CANONICAL]
        if (p, s) in committed and c["trace_sha256"] != committed[(p, s)]:
            problems.append(f"{p} seed {s}: canonical trace differs from the committed result")
        for m in (CANONICAL, I2RT, "yam_scrambled"):
            if r[m]["verdict"] != EXPECTED[p]:
                problems.append(f"{p} seed {s} on {m}: {r[m]['verdict']}, expected {EXPECTED[p]}")
            if r[m]["checks"] != c["checks"]:
                problems.append(f"{p} seed {s} on {m}: checks {r[m]['checks']} differ from canonical {c['checks']}")
        if not r[I2RT]["deterministic"]:
            problems.append(f"{p} seed {s} on {I2RT}: replays differently")
        if r["yam_scrambled"]["vs_canonical"]["point_m"] > SCRAMBLED_TOL_M:
            problems.append(f"{p} seed {s}: scrambled model drifts {r['yam_scrambled']['vs_canonical']['point_m']:.2e} m")
    return problems


def summary(rows: list[dict], seeds: list[int], check: dict, problems: list[str], committed: dict) -> str:
    models = (CANONICAL, I2RT, "yam_scrambled")
    lines = [
        "# The pack task on three YAM models",
        "",
        f"MuJoCo {mujoco.__version__}. The four reference policies plan in canonical joints on the canonical model;",
        "each model's joint map (`sim/yam_pack/joint_map.py`, configs in `sim/yam_pack/models/`) carries their",
        f"actions onto it. Seeds {seeds[0]}..{seeds[-1]} per policy. Produced by `python -m sim.yam_pack.compare_models`.",
        "",
        "- `menagerie_yam`: the canonical model every earlier result ran on (Robocurve's collision YAM).",
        "- `i2rt_yam_v1`: i2rt's own release, v1.4.0 (`fe78bae`), arm yam/v1 + LINEAR_4310 gripper.",
        "- `yam_scrambled`: the canonical arm with renamed joints, joints 2 and 5 reversed and five joints",
        "  re-zeroed (`sim/yam_pack/models/synthetic.py`), to show the map's arithmetic works where i2rt's needs none.",
        "",
        f"Result: **{'all verdicts and checks agree' if not problems else f'{len(problems)} disagreements'}**.",
        "",
        "| Policy | Expected | " + " | ".join(f"{m} PASS/FAIL" for m in models) + " | Checks equal to canonical | i2rt replays |",
        "|---|---|" + "---|" * len(models) + "---|---|",
    ]
    for p in POLICIES:
        rs = [r for r in rows if r["policy"] == p]
        if not rs:
            continue
        cells = []
        for m in models:
            n = sum(r[m]["verdict"] == "PASS" for r in rs)
            cells.append(f"{n} / {len(rs) - n}")
        same = sum(all(r[m]["checks"] == r[CANONICAL]["checks"] for m in models) for r in rs)
        det = sum(r[I2RT]["deterministic"] for r in rs)
        lines.append(f"| {p} | {EXPECTED[p]} | " + " | ".join(cells) + f" | {same}/{len(rs)} | {det}/{len(rs)} |")
    match = sum(r[CANONICAL]["trace_sha256"] == committed.get((r["policy"], r["seed"])) for r in rows)
    lines += ["", f"Canonical traces identical to the committed results: {match}/{len(rows)}.", "",
              "How far each model's trace sits from the canonical one (largest over all frames and named points):", "",
              "| Policy | i2rt point, mm | i2rt point less the 9.5 mm lift, mm | i2rt joint, rad | scrambled point, mm | i2rt clamped commands | i2rt time to stop L / R, ms |",
              "|---|---|---|---|---|---|---|"]
    for p in POLICIES:
        rs = [r for r in rows if r["policy"] == p]
        if not rs:
            continue
        ip = max(r[I2RT]["vs_canonical"]["point_m"] for r in rs) * 1000
        il = max(r[I2RT]["vs_canonical"]["point_after_lift_m"] for r in rs) * 1000
        ij = max(r[I2RT]["vs_canonical"]["joint_rad"] for r in rs)
        sp = max(r["yam_scrambled"]["vs_canonical"]["point_m"] for r in rs) * 1000
        clips = sum(r[I2RT]["limit_clips"] for r in rs)
        tts = {a: [r[I2RT]["time_to_stop_ms"][a] for r in rs] for a in ("left", "right")}
        fmt = {a: ("never" if any(x is None for x in v) else f"{max(v):g}") for a, v in tts.items()}
        lines.append(f"| {p} | {ip:.1f} | {il:.2f} | {ij:.4f} | {sp:.3f} | {clips} | {fmt['left']} / {fmt['right']} |")
    lines += ["", "All columns are maxima over every seed, frame and named point. The scrambled model is the same",
              "physical arm, and its kinematics match the canonical model's exactly under its map (`fit.check`); what",
              "is left is MuJoCo's constraint solver resolving joint friction and contacts in different joint",
              "coordinates (with joint friction off, the two agree to about 1e-9 m over the first 0.1 s). The bar is",
              f"{SCRAMBLED_TOL_M * 1000:g} mm.", "", "The i2rt column is the physical difference between the two descriptions of the arm: its shoulder",
              f"sits {check['lift_mm'][2]:g} mm higher (every axis line and named point moves by that, the same at every pose;",
              f"after removing it the axes agree to {max(check['axis_line_after_lift_mm']):g} mm and the named points to "
              f"{max(check['points_after_lift_mm'].values()):g} mm), and its",
              "servo response differs with its masses. Joint ranges where i2rt's model differs (radians, canonical convention):", ""]
    for j, r in check["range_differences_rad"].items():
        lines.append(f"- {j}: canonical {r['canonical']}, i2rt {r['this_model']}")
    if problems:
        lines += ["", "Disagreements:", ""] + [f"- {x}" for x in problems]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", choices=POLICIES)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    seeds = list(range(args.seeds))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        scrambled = synthetic.write(Path(tmp))
        jobs = [(p, s, str(scrambled)) for p in (args.policy or POLICIES) for s in seeds]
        with Pool(args.workers) as pool:
            rows = pool.map(job, jobs)
    committed = committed_hashes()
    problems = judge(rows, committed)
    check = fit.check(load_config(I2RT))
    for x in problems:
        print(x)
    print(f"{len(rows)} policy-seeds, {len(problems)} disagreements, {time.time() - t0:.0f} s")
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "comparison.json").write_text(json.dumps({"fit_check": check, "rows": rows}, indent=1, sort_keys=True) + "\n")
        (OUT / "SUMMARY.md").write_text(summary(rows, seeds, check, problems, committed))
    else:
        print(summary(rows, seeds, check, problems, committed))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
