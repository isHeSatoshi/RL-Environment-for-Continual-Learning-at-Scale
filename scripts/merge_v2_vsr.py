"""Merge the vsr_nogold recovery run back into runs/sccl_v2.

The main sccl_v2 ladder crashed at vsr_nogold Ep ~31/32 with OSError 28
(C: temp disk full). Five learners checkpointed to metrics_partial.json;
vsr_nogold was re-run alone (configs/sccl_v2_vsr_recovery.json, identical
stream + experiment). This script splices the completed learner row back
in and writes the final metrics.json that gcl.report reads.

Safety rails:
  - canary stream_hash of both runs must match (same stream identity)
  - canary must be clean in both
  - experiment configs must agree modulo out_dir (same regime)
  - refuses to overwrite an existing metrics.json without --force

Gold is not consumed here; this is bookkeeping on measurement artifacts.

Usage:
  python scripts/merge_v2_vsr.py [--main runs/sccl_v2] [--vsr runs/sccl_v2_vsr] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="runs/sccl_v2")
    ap.add_argument("--vsr", default="runs/sccl_v2_vsr")
    ap.add_argument("--learner", default="vsr_nogold")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    main_dir = os.path.join(REPO, args.main)
    vsr_dir = os.path.join(REPO, args.vsr)

    partial = _load(os.path.join(main_dir, "metrics_partial.json"))
    vsr = _load(os.path.join(vsr_dir, "metrics.json"))

    # ---- stream identity -----------------------------------------------------
    c1, c2 = partial["canary"], vsr["canary"]
    if c1["stream_hash"] != c2["stream_hash"]:
        print(f"ABORT: stream_hash mismatch {c1['stream_hash']} vs {c2['stream_hash']}")
        return 1
    if not (c1.get("clean") and c2.get("clean")):
        print("ABORT: canary not clean in one of the runs")
        return 1

    # ---- regime identity (config must agree modulo out_dir) -------------------
    # Normalize through the current ExperimentConfig first: runs checkpointed
    # under older code lack newer fields (sccl_nbhd_*, torch_seed), and raw
    # dict comparison would flag the missing keys as regime drift.
    sys.path.insert(0, REPO)
    import dataclasses
    from gcl.config import ExperimentConfig
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}

    def _norm(cfg: dict) -> dict:
        known = {k: v for k, v in cfg.items() if k in fields}
        d = dataclasses.asdict(ExperimentConfig(**known))
        d.pop("out_dir", None)
        return d

    e1, e2 = _norm(partial["config"]), _norm(vsr["config"])
    diffs = {k: (e1.get(k), e2.get(k)) for k in set(e1) | set(e2) if e1.get(k) != e2.get(k)}
    if diffs:
        print("ABORT: experiment configs differ beyond out_dir:")
        for k, (a, b) in sorted(diffs.items()):
            print(f"  {k}: {a!r} vs {b!r}")
        return 1

    name = args.learner
    if name not in vsr["learners"]:
        print(f"ABORT: learner {name!r} not in recovery run")
        return 1
    if name in partial["learners"]:
        print(f"NOTE: {name!r} already present in partial — replacing with completed row")

    # ---- splice ---------------------------------------------------------------
    merged = dict(partial)
    merged["learners"] = dict(partial["learners"])
    merged["learners"][name] = vsr["learners"][name]

    out_path = os.path.join(main_dir, "metrics.json")
    if os.path.exists(out_path) and not args.force:
        print(f"ABORT: {out_path} exists (use --force to overwrite)")
        return 1
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {out_path} ({len(merged['learners'])} learners)")

    # ---- artifacts: trajectory + final adapters (for retro eval) --------------
    traj_src = os.path.join(vsr_dir, f"trajectories_{name}.jsonl")
    if os.path.exists(traj_src):
        shutil.copy2(traj_src, os.path.join(main_dir, f"trajectories_{name}.jsonl"))
        print(f"copied trajectories_{name}.jsonl")
    ad_src = os.path.join(vsr_dir, f"adapters_{name}")
    if os.path.isdir(ad_src):
        ad_dst = os.path.join(main_dir, f"adapters_{name}")
        if os.path.isdir(ad_dst):
            shutil.rmtree(ad_dst)  # crashed partial registry — replace wholesale
        shutil.copytree(ad_src, ad_dst)
        print(f"replaced adapters_{name}/ from recovery run")

    r = merged["learners"][name]["report"]
    print(f"merged row: {name} acc={r['acc']:.3f} bwt={r['bwt']:+.3f} "
          f"forget={r['forgetting']:.3f} updates={merged['learners'][name]['updates']}"
          f"/rollbacks={merged['learners'][name]['rollbacks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
