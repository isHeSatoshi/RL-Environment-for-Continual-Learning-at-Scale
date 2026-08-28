"""v8 (Branch F, G1) follow-up: 3-seed confirmation of the generalization-
witness breakthrough cell.

Pre-registration (RESEARCH_NOTES_v5.md "BRANCH F PRE-REGISTRATION" +
configs/sccl_v8.json): if the seed-42 v8 verdict is BREAKTHROUGH
(sccl_genprobe_strict_anchor.arith >= 0.55 AND frontier >= sccl - 0.02 AND
accepted updates >= 5, as checked by scripts/v8_check.py), the result must be
confirmed at seeds 43/44 BEFORE any headline claim — same protocol as
v3/v5/v5b/v6/v7 (seeded per learner as torch_seed + learner index, so a
learner at the same position draws the same relative seed across runs).

Do NOT run this script unless scripts/v8_check.py reports BREAKTHROUGH: PASS
and all fail-closed checks (C1-C5) passed, with the anchor row not degenerate.

This script:
  1. snapshots the completed seed-42 ladder (runs/sccl_v8) to
     runs/sccl_v8_s42 so all three seeds share one directory layout;
  2. runs the identical ladder at torch_seed 43 and 44 via run_seeds.py
     (sequential subprocesses, fresh GPU state per seed, logs in
     runs/sccl_v8_s{43,44}_log.txt);
  3. re-aggregates all three seeds into runs/sccl_v8_seeds/aggregate.json
     under the usual rails (single stream hash, clean canaries).

Run only AFTER runs/sccl_v8/metrics.json is final.
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
CONFIG = os.path.join(REPO, "configs", "sccl_v8.json")
S42_SRC = os.path.join(REPO, "runs", "sccl_v8")
S42_DST = os.path.join(REPO, "runs", "sccl_v8_s42")
FAMILIES = ["arith", "math_word", "string", "drift"]
ROWS = ["sccl", "sccl_capprobe_strat", "sccl_genprobe",
        "sccl_genprobe_strict", "sccl_genprobe_strict_anchor"]
BREAKTHROUGH_CELL = "sccl_genprobe_strict_anchor"
DEGENERATE_MIN_UPDATES = 5


def fam_scores(learner: dict) -> dict:
    fh = learner.get("eval_detail", {}).get("final_heldout", {})
    out = {}
    for fam in FAMILIES:
        tasks = fh.get(fam, [])
        out[fam] = sum(t["score"] for t in tasks) / max(1, len(tasks))
    return out


def arith_table(seeds: list[int]) -> None:
    """Per-seed arith holdout (the pre-registered breakthrough axis)."""
    base = json.load(open(CONFIG)).get("out_dir", "runs/sccl_v8")
    per: dict[str, dict[int, float]] = {}
    front: dict[str, dict[int, float]] = {}
    upd: dict[str, dict[int, int]] = {}
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
            upd.setdefault(name, {})[seed] = int(
                learners[name].get("report", {}).get("updates", 0))

    def ms(xs: list[float]):
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
        return mu, math.sqrt(var)

    print("\n=== Per-seed ARITH holdout (breakthrough axis) ===")
    hdr = f"{'learner':26s}" + "".join(f"  s{s}" for s in seeds) + "   mean+/-std"
    print(hdr)
    for name in ROWS:
        if name not in per:
            continue
        vals = [per[name].get(s) for s in seeds]
        got = [v for v in vals if v is not None]
        mu, sd = ms(got) if len(got) > 1 else (got[0] if got else float("nan"), 0.0)
        cells = "".join(f"  {per[name][s]:.3f}" if s in per[name] else "    -  "
                        for s in seeds)
        print(f"{name:26s}{cells}   {mu:.3f}+/-{sd:.3f}")
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
        print(f"{name:26s}{cells}   {mu:+.3f}+/-{sd:.3f}")
    print("\n=== Per-seed accepted updates (anti-stall) ===")
    print(hdr)
    for name in ROWS:
        if name not in upd:
            continue
        cells = "".join(f"  {upd[name][s]:>4d}" if s in upd[name] else "     -"
                        for s in seeds)
        print(f"{name:26s}{cells}")

    if per.get(BREAKTHROUGH_CELL) and front.get("sccl") \
            and front.get(BREAKTHROUGH_CELL):
        # paired per-seed breakthrough check (pre-registration rule):
        # arith >= 0.55 AND frontier >= sccl - 0.02 AND updates >= 5
        ok = 0
        for s in seeds:
            if s in per.get(BREAKTHROUGH_CELL, {}) and s in front.get("sccl", {}) \
                    and s in front.get(BREAKTHROUGH_CELL, {}):
                if per[BREAKTHROUGH_CELL][s] >= 0.55 and \
                        front[BREAKTHROUGH_CELL][s] >= front["sccl"][s] - 0.02 and \
                        upd.get(BREAKTHROUGH_CELL, {}).get(s, 0) >= DEGENERATE_MIN_UPDATES:
                    ok += 1
        print(f"\nBREAKTHROUGH rule met on {ok}/{len(seeds)} seeds "
              f"(arith>=0.55 AND frontier>=sccl-0.02 AND updates>=5, "
              f"paired per seed)")


def main() -> None:
    final = os.path.join(S42_SRC, "metrics.json")
    if not os.path.exists(final):
        print(f"[v8seeds] {final} missing — the seed-42 ladder is not final; aborting")
        sys.exit(1)
    if not os.path.exists(S42_DST):
        print(f"[v8seeds] snapshotting seed-42 ladder -> {S42_DST}")
        shutil.copytree(S42_SRC, S42_DST)
    else:
        print(f"[v8seeds] {S42_DST} already present — reusing snapshot")

    # run seeds 43/44 (run_seeds aggregates its own seed list first)...
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_seeds.py"),
                        "--config", CONFIG, "--seeds", "43,44"], cwd=REPO)
    if r.returncode != 0:
        print(f"[v8seeds] seed runs failed (exit {r.returncode}); not aggregating")
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
