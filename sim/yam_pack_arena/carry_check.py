"""Does the Isaac grasp (kinematic carry) change any verdict? Runs every policy and seed of the MuJoCo
results twice in MuJoCo, once with the weld (``scene.World``) and once through ``core.PackDriver``
with the carry, scores both with the oracle and writes the comparison.

    python -m sim.yam_pack_arena.carry_check            # 4 policies x 20 seeds, about 8 minutes
    -> results/sim-track/yam-pack-arena/carry_vs_weld.json, CARRY.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sim.yam_pack.oracle import score
from sim.yam_pack.run import ENVELOPE, EXPECTED, episode as weld_episode
from sim.yam_pack.spec import load_spec
from sim.yam_pack_arena.core import episode

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sim-track/yam-pack-arena"


def compare(policy: str, seed: int, spec: dict, envelope: dict) -> dict:
    tc, _ = episode(policy, seed, spec)
    tw, _ = weld_episode(policy, seed, spec)
    v, w = score(tc, envelope, spec), score(tw, envelope, spec)
    n = min(len(tc), len(tw))
    point_gap = max(float(np.linalg.norm(np.array(a["points"][p]) - np.array(b["points"][p])))
                    for a, b in zip(tc[:n], tw[:n]) for p in a["points"])
    obj_gap = max(float(np.linalg.norm(np.array(tc[-1]["objects"][k]) - np.array(tw[-1]["objects"][k])))
                  for k in tc[-1]["objects"])
    kinds = lambda r: sorted({x.get("kind", "?") if isinstance(x, dict) else str(x) for x in r["violations"]})  # noqa: E731
    return {"policy": policy, "seed": seed, "carry": v["verdict"], "weld": w["verdict"],
            "checks_carry": {k: c["result"] for k, c in v["checks"].items()},
            "checks_weld": {k: c["result"] for k, c in w["checks"].items()},
            "violation_kinds_carry": kinds(v), "violation_kinds_weld": kinds(w),
            "max_point_gap_m": round(point_gap, 6), "final_object_gap_m": round(obj_gap, 6),
            "frames": [len(tc), len(tw)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    spec, envelope = load_spec(), json.loads(ENVELOPE.read_text())
    rows = []
    for pol in EXPECTED:
        for s in range(args.seeds):
            r = compare(pol, s, spec, envelope)
            rows.append(r)
            print(pol, s, r["carry"], r["weld"], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "carry_vs_weld.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Carry vs weld in MuJoCo",
             "",
             "Isaac Lab has no runtime weld, so the port carries a grasped object kinematically",
             "(sim/yam_pack_arena/core.py). This runs both grasps in MuJoCo on every committed seed.",
             "",
             "| Policy | Episodes | Same verdict | Same checks | Same violation kinds | Max named-point gap (mm) | Max final object gap (mm) |",
             "|---|---|---|---|---|---|---|"]
    for pol in EXPECTED:
        rs = [r for r in rows if r["policy"] == pol]
        lines.append(f"| {pol} | {len(rs)} | {sum(r['carry'] == r['weld'] for r in rs)} | "
                     f"{sum(r['checks_carry'] == r['checks_weld'] for r in rs)} | "
                     f"{sum(r['violation_kinds_carry'] == r['violation_kinds_weld'] for r in rs)} | "
                     f"{1000 * max(r['max_point_gap_m'] for r in rs):.1f} | {1000 * max(r['final_object_gap_m'] for r in rs):.1f} |")
    lines += ["", "Final object gap is where blocks come to rest; a block can settle a few centimetres",
              "differently in the box without changing any check.", ""]
    (args.out / "CARRY.md").write_text("\n".join(lines))
    same = all(r["carry"] == r["weld"] and r["checks_carry"] == r["checks_weld"] for r in rows)
    print("all verdicts and checks the same" if same else "DIFFERENCES: see carry_vs_weld.json")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
