"""v6 (Branch D) follow-up: 3-seed confirmation of the ensemble + anchor ladder.

Pre-registration (RESEARCH_NOTES_v5.md "BRANCH D PRE-REGISTRATION" +
configs/sccl_v6.json): if the seed-42 v6 verdict is BREAKTHROUGH
(sccl_capens_anchor.arith >= 0.55 AND frontier >= sccl - 0.02, as checked by
scripts/v6_check.py), the result must be confirmed at seeds 43/44 BEFORE any
headline claim — same protocol as v3/v5/v5b (seeded per learner as torch_seed +
learner index, so a learner at the same position draws the same relative seed
across runs).

Do NOT run this script unless scripts/v6_check.py reports BREAKTHROUGH: PASS
and all fail-closed checks (C1-C5) passed.

This script:
  1. snapshots the completed seed-42 ladder (runs/sccl_v6) to
     runs/sccl_v6_s42 so all three seeds share one directory layout;
  2. runs the identical ladder at torch_seed 43 and 44 via run_seeds.py
     (sequential subprocesses, fresh GPU state per seed, logs in
     runs/sccl_v6_s{43,44}_log.txt);
  3. re-aggregates all three seeds into runs/sccl_v6_seeds/aggregate.json
     under the usual rails (single stream hash, clean canaries).

Run only AFTER runs/sccl_v6/metrics.json is final.
Launch convention (Windows): TMPDIR/TEMP/TMP -> /d/gcl_tmp.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "configs", "sccl_v6.json")
S42_SRC = os.path.join(REPO, "runs", "sccl_v6")
S42_DST = os.path.join(REPO, "runs", "sccl_v6_s42")


def main() -> None:
    final = os.path.join(S42_SRC, "metrics.json")
    if not os.path.exists(final):
        print(f"[v6seeds] {final} missing — the seed-42 ladder is not final; aborting")
        sys.exit(1)
    if not os.path.exists(S42_DST):
        print(f"[v6seeds] snapshotting seed-42 ladder -> {S42_DST}")
        shutil.copytree(S42_SRC, S42_DST)
    else:
        print(f"[v6seeds] {S42_DST} already present — reusing snapshot")

    # run seeds 43/44 (run_seeds aggregates its own seed list first)...
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_seeds.py"),
                        "--config", CONFIG, "--seeds", "43,44"], cwd=REPO)
    if r.returncode != 0:
        print(f"[v6seeds] seed runs failed (exit {r.returncode}); not aggregating")
        sys.exit(r.returncode)

    # ...then re-aggregate with the seed-42 snapshot included.
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "run_seeds.py"),
                        "--config", CONFIG, "--seeds", "42,43,44", "--aggregate-only"],
                       cwd=REPO)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
