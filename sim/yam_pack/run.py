"""Run the pack-with-hand task in MuJoCo and score every episode with the oracle.

    python -m sim.yam_pack.run                         # 20 seeds x 4 policies, determinism check, summary
    python -m sim.yam_pack.run --policy comply --seeds 1 --trace out.json.gz

Each episode runs twice from the same seed; the two traces and the two verdicts must hash the same.
Results go to results/sim-track/yam-pack/ (one JSON per policy, plus SUMMARY.md).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import mujoco

from sim.yam_pack.oracle import score, sha256_json
from sim.yam_pack.policies import POLICIES, Packer
from sim.yam_pack.scene import World
from sim.yam_pack.spec import load_spec

ROOT = Path(__file__).resolve().parents[2]
ENVELOPE = ROOT / "embodiments/abc-bimanual/envelope.json"
OUT = ROOT / "results/sim-track/yam-pack"
EXPECTED = {"comply": "PASS", "ignore": "FAIL", "right_only": "FAIL", "do_nothing": "FAIL"}


def episode(policy: str, seed: int, spec: dict | None = None, overrides: dict | None = None) -> tuple[list[dict], dict]:
    spec = spec or load_spec()
    w = World(spec, seed, overrides)
    pol = Packer(w, policy)
    settle = 1.0  # keep stepping after the plan ends so the last block lands
    end_at = None
    while w.t < spec["episode"]["max_time"]:
        w.step(pol.act(w.observation()))
        if end_at is None and pol.done() and w.hand_done:
            end_at = w.t + settle
        if end_at is not None and w.t >= end_at:
            break
    return w.trace, {"seed": seed, "policy": policy, "first_release_t": w.first_release_t,
                     "hand_target": w.hand_target.tolist(), "hand_arrive_s": w.hand_arrive,
                     "block_xy": w.block_xy}


def run(policy: str, seed: int, envelope: dict, spec: dict) -> dict:
    tr1, info = episode(policy, seed, spec)
    tr2, _ = episode(policy, seed, spec)
    v1, v2 = score(tr1, envelope, spec), score(tr2, envelope, spec)
    return {**info, "verdict": v1["verdict"], "checks": {k: c["result"] for k, c in v1["checks"].items()},
            "why": {k: c["why"] for k, c in v1["checks"].items()},
            "violations": v1["violations"], "metrics": v1["metrics"],
            "trace_sha256": sha256_json(tr1),
            "deterministic": sha256_json(tr1) == sha256_json(tr2) and v1["verdict_sha256"] == v2["verdict_sha256"]}


def summary(results: dict, seeds: list[int], spec: dict) -> str:
    lines = [
        "# Sim track: pack with a hand mid-task (B3, bimanual) in MuJoCo",
        "",
        f"MuJoCo {mujoco.__version__}, Robocurve's YAM collision model (inspect-robots-yam 9b5ebb5, Menagerie 71f066a),",
        f"two arms at y = +/-0.30 m, {spec['robot']['control_hz']:g} Hz 14-D joint-position actions, physics at "
        f"{spec['robot']['physics_dt'] * 1000:g} ms. Envelope: embodiments/abc-bimanual/envelope.json (placeholder numbers).",
        f"Seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} per policy); every episode run twice from its seed.",
        "",
        "| Policy | Expected | PASS | FAIL | Deterministic | Violation events (mean) | Violations per metre (mean) | Time to stop L / R, ms (max) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, rows in results.items():
        npass = sum(r["verdict"] == "PASS" for r in rows)
        det = sum(r["deterministic"] for r in rows)
        ev = sum(r["metrics"]["violation_events"] for r in rows) / len(rows)
        vpm = [r["metrics"]["violations_per_metre"] for r in rows if r["metrics"]["violations_per_metre"] is not None]
        tts = {a: [r["metrics"]["time_to_stop_ms"][a] for r in rows] for a in ("left", "right")}
        fmt = {a: ("never" if any(x is None for x in v) else f"{max(v):g}") for a, v in tts.items()}
        lines.append(f"| {p} | {EXPECTED[p]} | {npass} | {len(rows) - npass} | {det}/{len(rows)} | {ev:.1f} | "
                     f"{(sum(vpm) / len(vpm)) if vpm else 0:.2f} | {fmt['left']} / {fmt['right']} |")
    lines += ["", "Failing checks and violation kinds, per policy:", ""]
    for p, rows in results.items():
        failing: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for r in rows:
            for k, ok in r["checks"].items():
                if not ok:
                    failing[k] = failing.get(k, 0) + 1
            for k in r["violations"]:
                kinds[k] = kinds.get(k, 0) + 1
        lines.append(f"- **{p}**: failing checks {failing or 'none'}; violations (episodes) {kinds or 'none'}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", choices=POLICIES)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--trace", type=Path, help="write the first episode's trace (gzip JSON) here")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write results/")
    args = ap.parse_args(argv)
    spec = load_spec()
    envelope = json.loads(ENVELOPE.read_text())
    seeds = list(range(args.seeds))
    results, ok = {}, True
    t0 = time.time()
    for p in args.policy or POLICIES:
        results[p] = [run(p, s, envelope, spec) for s in seeds]
        bad = [r["seed"] for r in results[p] if r["verdict"] != EXPECTED[p] or not r["deterministic"]]
        ok &= not bad
        print(f"{p:11s} {sum(r['verdict'] == 'PASS' for r in results[p])}/{len(seeds)} PASS, "
              f"expected {EXPECTED[p]}, unexpected or non-deterministic seeds: {bad or 'none'}")
    print(f"{time.time() - t0:.0f} s")
    if args.trace:
        tr, _ = episode((args.policy or POLICIES)[0], seeds[0], spec)
        with gzip.open(args.trace, "wt") as f:
            json.dump(tr, f)
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        for p, rows in results.items():
            (OUT / f"{p}.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
        (OUT / "SUMMARY.md").write_text(summary(results, seeds, spec))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
