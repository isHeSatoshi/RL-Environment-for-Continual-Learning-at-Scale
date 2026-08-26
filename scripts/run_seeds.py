"""Multi-seed ladder runner + aggregator.

Single-run SCCL comparisons are statistically weak: the certification loop is
stochastic (temperature-sampled self-tests and candidate pools), and the
unseeded v1 sccl learner scored 0.637 on the original run vs 0.744 on the v2
rerun (same stream hash). For paper-grade numbers we run the same ladder under
several torch_seeds and report mean +/- std.

This tool:
  1. rewrites the config for each seed (experiment.torch_seed = seed,
     out_dir = <out_base>_s<seed>), writes a temp config, and runs
     `python -m gcl.runner` as a sequential subprocess (fresh GPU state per
     seed run), logging to <out_base>_s<seed>_log.txt;
  2. after all seeds finish, aggregates each learner's metrics across seeds
     (mean +/- std) into <out_base>_seeds/aggregate.json and prints a table.

Usage:
  python scripts/run_seeds.py --config configs/sccl_v3.json --seeds 42,43,44
  python scripts/run_seeds.py --config configs/sccl_v3.json --seeds 42,43,44 --aggregate-only
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_seeds(config_path: str, seeds: list[int], keep_tmp: bool = False) -> list[str]:
    base = _load(config_path)
    base_out = base.get("out_dir", "runs/seeds")
    out_dirs = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg.setdefault("experiment", {})["torch_seed"] = int(seed)
        out_dir = f"{base_out}_s{seed}"
        cfg["out_dir"] = out_dir
        cfg["experiment"]["out_dir"] = out_dir
        tmp_cfg = os.path.join(REPO, "runs", f"_tmp_seedcfg_s{seed}.json")
        os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
        with open(tmp_cfg, "w") as f:
            json.dump(cfg, f, indent=2)
        log_path = f"{out_dir}_log.txt"
        print(f"[seeds] seed={seed} -> {out_dir} (log {log_path})", flush=True)
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.run([sys.executable, "-m", "gcl.runner", "--config", tmp_cfg],
                                  cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60.0
        if proc.returncode != 0:
            print(f"[seeds] seed={seed} FAILED (exit {proc.returncode}) after {dt:.1f} min — see {log_path}",
                  flush=True)
            raise RuntimeError(f"seed run {seed} failed")
        print(f"[seeds] seed={seed} done in {dt:.1f} min", flush=True)
        out_dirs.append(out_dir)
        if not keep_tmp:
            try:
                os.remove(tmp_cfg)
            except OSError:
                pass
    return out_dirs


def _mean_std(xs: list[float]):
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs) if len(xs) > 1 else 0.0
    return mu, math.sqrt(var)


def aggregate(config_path: str, seeds: list[int]) -> dict:
    base = _load(config_path)
    base_out = base.get("out_dir", "runs/seeds")
    agg_dir = f"{base_out}_seeds"
    os.makedirs(agg_dir, exist_ok=True)
    learners: dict[str, dict[str, list[float]]] = {}
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        mp = os.path.join(f"{base_out}_s{seed}", "metrics.json")
        if not os.path.exists(mp):
            print(f"[seeds] missing {mp} — skipping seed {seed}")
            continue
        m = _load(mp)
        per_seed[str(seed)] = {
            "stream_hash": m.get("canary", {}).get("stream_hash", ""),
            "learners": {},
        }
        for name, d in m.get("learners", {}).items():
            r = d.get("report", {})
            row = {"acc": r.get("acc", 0.0), "bwt": r.get("bwt", 0.0),
                   "forgetting": r.get("forgetting", 0.0), "fwt": r.get("fwt", 0.0),
                   "frontier": d.get("frontier_score", 0.0),
                   "updates": d.get("updates", 0), "rollbacks": d.get("rollbacks", 0)}
            per_seed[str(seed)]["learners"][name] = row
            slot = learners.setdefault(name, {k: [] for k in row})
            for k, v in row.items():
                slot[k].append(float(v))
    summary: dict[str, dict] = {}
    print(f"\n=== Seed aggregate ({len(per_seed)} seeds) -> {agg_dir} ===")
    print(f"{'Learner':<14} {'ACC':>14} {'Forget':>14} {'BWT':>14} {'Frontier':>14}")
    for name, cols in learners.items():
        row = {}
        cells = []
        for k in ("acc", "forgetting", "bwt", "frontier"):
            mu, sd = _mean_std(cols[k])
            row[k] = {"mean": mu, "std": sd}
            cells.append(f"{mu:.3f}+/-{sd:.3f}")
        mu_u, _ = _mean_std(cols["updates"])
        mu_r, _ = _mean_std(cols["rollbacks"])
        row["updates"] = {"mean": mu_u}
        row["rollbacks"] = {"mean": mu_r}
        summary[name] = row
        print(f"{name:<14} " + " ".join(f"{c:>14}" for c in cells))
    out = {"config": config_path, "seeds": seeds, "per_seed": per_seed, "summary": summary}
    with open(os.path.join(agg_dir, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[seeds] wrote {os.path.join(agg_dir, 'aggregate.json')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip running; aggregate existing <out>_s<seed> dirs")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not args.aggregate_only:
        run_seeds(args.config, seeds, keep_tmp=args.keep_tmp)
    aggregate(args.config, seeds)


if __name__ == "__main__":
    main()
