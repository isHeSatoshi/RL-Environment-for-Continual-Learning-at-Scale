"""v7 (Branch E) follow-up: 3-seed confirmation of the E1 pass-rate ladder.

Pre-registration (RESEARCH_NOTES_v5.md "BRANCH E PRE-REGISTRATION" +
configs/sccl_v7.json): if the seed-42 v7 verdict is BREAKTHROUGH
(sccl_ens_strict_anchor.arith >= 0.55 AND frontier >= sccl - 0.02, as checked
by scripts/v7_check.py), the result must be confirmed at seeds 43/44 BEFORE
any headline claim — same protocol as v3/v5/v5b/v6 (seeded per learner as
torch_seed + learner index, so a learner at the same position draws the same
relative seed across runs).

Do NOT run this script unless scripts/v7_check.py reports BREAKTHROUGH: PASS
and all fail-closed checks (C1-C5) passed.

This script:
  1. snapshots the completed seed-42 ladder (runs/sccl_v7) to
     runs/sccl_v7_s42 so all three seeds share one directory layout;
  2. runs the identical ladder at torch_seed 43 and 44 via run_seeds.py
     (sequential subprocesses, fresh GPU state per seed, logs in
     runs/sccl_v7_s{43,44}_log.txt);
  3. re-aggregates all three seeds into runs/sccl_v7_seeds/aggregate.json
     under the usual rails (single stream hash, clean canaries).

Run only AFTER runs/sccl_v7/metrics.json is final.
Launch convention (Windows): TMPDIR/TEMP/TMP -> /d/gcl_tmp.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "sccl_v7.json")
S42_SRC = os.path.join(REPO, "runs", "sccl_v7")
S42_DST = os.path.join(REPO, "runs", "sccl_v7_s42")
FAMILIES = ["arith", "math_word", "string", "drift"]
ROWS = ["sccl", "sccl_capprobe_strat", "sccl_strict", "sccl_majority",
        "sccl_ens_strict", "sccl_ens_strict_anchor"]
BREAKTHROUGH_CELL = "sccl_ens_strict_anchor"


def fam_scores(learner: dict) -> dict:
    fh = learner.get("eval_detail", {}).get("final_heldout", {})
    out = {}
    for fam in FAMILIES:
        tasks = fh.get(fam, [])
        out[fam] = sum(t["score"] for t in tasks) / max(1, len(tasks))
    return out


def arith_table(seeds: list[int]) -> None:
    """Per-seed arith holdout (the pre-registered breakthrough axis)."""
    base = json.load(open(CONFIG)).get("out_dir", "runs/sccl_v7")
    per: dict[str, dict[int, float]] = {}
    front: dict[str, dict[int, float]] = {}
    for seed in seeds:
        mp = os.path.join(REPO, f"{base}_s{seed}", "metrics.json")
        if not os.path.exists(mp):
            continue
        learners = json.load(open(mp))["learners"]
        for name in ROWS:
            if name not in learners:
                continue
            per.setdefault(name, {})[seed] = fam_scores(learners[name])["arith"]
            front.setdefault(name, {})[seed] = learners[name].get("frontier_score", 0.0)

    def ms(xs: list[float]):
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
        return mu, math.sqrt(var)

    print("\n=== Per-seed ARITH holdout (breakthrough axis) ===")
    hdr = f"{'learner':22s}" + "".join(f"  s{s}" for s in seeds) + "   mean+/-std"
    print(hdr)
    for name in ROWS:
        if name not in per:
            continue
        vals = [per[name].get(s) for s in seeds]
        got = [v for v in vals if v is not None]
        mu, sd = ms(got) if len(got) > 1 else (got[0] if got else float("nan"), 0.0)
        cells = "".join(f"  {per[name][s]:.3f}" if s in per[name] else "    -  "
                        for s in seeds)
        print(f"{name:22s}{cells}   {mu:.3f}+/-{sd:.3f}")
    print("\n=== Per-seed frontier ===")
    print(hdr)
    for name in ROWS:
        if name not in front:
            continue
        vals = [front[name].get(s) for s in seeds]
        got = [v for v in vals if v is not None]
        mu, sd = ms(got) if len(got) > 1 else (got[0] if got else float("nan"), 0.0)
        cells = "".join(f"  {front[name][s]:+.3f}" if s in front[name] else "    -  "
                        for s in seeds)
        print(f"{name:22s}{cells}   {mu:+.3f}+/-{sd:.3f}")
    ea = [per[BREAKTHROUGH_CELL][s] for s in seeds
          if s in per.get(BREAKTHROUGH_CELL, {})]
    sc = [front["sccl"][s] for s in seeds if s in front.get("sccl", {})]
    eaf = [front[BREAKTHROUGH_CELL][s] for s in seeds
           if s in front.get(BREAKTHROUGH_CELL, {})]
    if ea and sc and eaf:
        # paired per-seed breakthrough check
        ok = 0
        for s in seeds:
            if s in per.get(BREAKTHROUGH_CELL, {}) and s in front.get("sccl", {}) \
                    and s in front.get(BREAKTHROUGH_CELL, {}):
                if per[BREAKTHROUGH_CELL][s] >= 0.55 and \
                        front[BREAKTHROUGH_CELL][s] >= front["sccl"][s] - 0.02:
                    ok += 1
        print(f"\nBREAKTHROUGH rule met on {ok}/{len(seeds)} seeds "
              f"(arith>=0.55 AND frontier>=sccl-0.02, paired per seed)")


def main() -> None:
    final = os.path.join(S42_SRC, "metrics.json")
    if not os.path.exists(final):
        print(f"[v7seeds] {final} missing — the seed-42 ladder is not final; aborting")
        sys.exit(1)
    if not os.path.exists(S42_DST):
        print(f"[v7seeds] snapshotting seed-42 ladder -> {S42_DST}")
        shutil.copytree(S42_SRC, S42_DST)
    else:
        print(f"[v7seeds] {S42_DST} already present — reusing snapshot")

    # run seeds 43/44 (run_seeds aggregates its own seed list first)...
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_seeds.py"),
                        "--config", CONFIG, "--seeds", "43,44"], cwd=REPO)
    if r.returncode != 0:
        print(f"[v7seeds] seed runs failed (exit {r.returncode}); not aggregating")
        sys.exit(r.returncode)

    # ...then re-aggregate with the seed-42 snapshot included.
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_seeds.py"),
                        "--config", CONFIG, "--seeds", "42,43,44", "--aggregate-only"],
                       cwd=REPO)
    if r.returncode != 0:
        sys.exit(r.returncode)
    arith_table([42, 43, 44])


if __name__ == "__main__":
    main()
