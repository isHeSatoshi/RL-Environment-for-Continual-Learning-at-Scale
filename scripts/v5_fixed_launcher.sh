#!/usr/bin/env bash
# Poll until the FIRST v5 factorial flight (runs/sccl_v5) is final, then launch
# the CORRECTED rerun (configs/sccl_v5_fixed.json) with the engagement-tested
# anchor of commit e53923a. Launched detached:
#   nohup bash scripts/v5_fixed_launcher.sh > runs/sccl_v5_fixed_launcher.log 2>&1 &
# The first flight ran with the inert (pre-fix) anchor; its anchor cells are
# seed controls. The corrected rerun must bit-reproduce its non-anchor rows
# (frozen/sccl/sccl_strat/vsr_nogold) as the determinism check across the fix.
#
# DOUBLE-LAUNCH GUARD (post-mortem in RESEARCH_NOTES_v5.md): two copies of
# this wrapper once fired concurrent runners into the same out_dir and one
# segfaulted under GPU contention. Refuse to start if another copy holds the
# lock, or if the corrected run already produced artifacts.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v5_fixed_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v5fixed] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v5fixed] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -f runs/sccl_v5_fixed/metrics.json ]; then
  echo "[v5fixed] ABORT: runs/sccl_v5_fixed/metrics.json already exists (run complete)"
  exit 1
fi
if [ -f runs/sccl_v5_fixed/metrics_partial.json ]; then
  echo "[v5fixed] ABORT: runs/sccl_v5_fixed/metrics_partial.json exists (run in progress)"
  exit 1
fi

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
