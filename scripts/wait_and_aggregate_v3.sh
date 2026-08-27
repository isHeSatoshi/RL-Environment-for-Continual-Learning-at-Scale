#!/usr/bin/env bash
# Wait for the multi-seed v3 run (seeds 43,44) to finish, then aggregate
# 42/43/44 and emit paper/results_v3.tex. Runs detached so the agent is not
# stuck busy-polling the GPU run. Writes a WATCHER-DONE marker on completion.
set -uo pipefail
cd "$(dirname "$0")/.."

export TMPDIR=/d/gcl_tmp TEMP=/d/gcl_tmp TMP=/d/gcl_tmp
export PYTHONUNBUFFERED=1

M44=runs/sccl_v3_s44/metrics.json
MARKER=runs/sccl_v3_seeds_watchdone.txt

echo "[watcher] waiting for $M44 ..."
while true; do
  if [ -f "$M44" ]; then
    # confirm it is a fully-written, valid metrics file with learners
    if python - "$M44" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert "learners" in m and len(m["learners"])>=5, "incomplete"
print("valid")
PY
    then
      echo "[watcher] seed-44 metrics present and valid; grace period for orchestrator exit..."
      sleep 90
      break
    fi
  fi
  sleep 120
done

echo "[watcher] aggregating seeds 42,43,44 ..."
python scripts/run_seeds.py --config configs/sccl_v3.json --seeds 42,43,44 --aggregate-only || { echo "WATCHER-AGG-FAIL"; exit 1; }

echo "[watcher] generating paper/results_v3.tex ..."
python -m gcl.report --aggregate runs/sccl_v3_seeds --out paper/results_v3.tex --prefix vthree || { echo "WATCHER-REPORT-FAIL"; exit 1; }

echo "WATCHER-DONE $(date)" | tee "$MARKER"
