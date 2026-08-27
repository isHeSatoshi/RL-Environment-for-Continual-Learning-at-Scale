#!/usr/bin/env bash
# Poll until the FIRST v5 factorial flight (runs/sccl_v5) is final, then launch
# the CORRECTED rerun (configs/sccl_v5_fixed.json) with the engagement-tested
# anchor of commit e53923a. Launched detached:
#   nohup bash scripts/v5_fixed_launcher.sh > runs/sccl_v5_fixed_launcher.log 2>&1 &
# The first flight ran with the inert (pre-fix) anchor; its anchor cells are
# seed controls. The corrected rerun must bit-reproduce its non-anchor rows
# (frozen/sccl/sccl_strat/vsr_nogold) as the determinism check across the fix.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

echo "[v5fixed] $(date '+%F %T') waiting for runs/sccl_v5/metrics.json ..."
while [ ! -f runs/sccl_v5/metrics.json ]; do
  sleep 120
done
echo "[v5fixed] $(date '+%F %T') first flight final; settling 120s for GPU release"
sleep 120

echo "[v5fixed] $(date '+%F %T') launching corrected factorial (configs/sccl_v5_fixed.json)"
python -m gcl.runner --config configs/sccl_v5_fixed.json
rc=$?
echo "[v5fixed] $(date '+%F %T') corrected factorial exited rc=$rc"
exit $rc
