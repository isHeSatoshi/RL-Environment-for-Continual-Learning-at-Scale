#!/usr/bin/env bash
# Wait for the v8 (Branch F, G1 generalization witnesses) ladder (runs/sccl_v8)
# to finalize, then run the pre-registered verification (fail-closed C1-C5 +
# H1/H2/H3/M1/BREAKTHROUGH + degeneracy guard). metrics.json is written only
# at run completion; during the run only metrics_partial.json exists.
set -u
cd "$(dirname "$0")/.."
METRICS=runs/sccl_v8/metrics.json
echo "[v8-check-watcher] $(date '+%F %T') waiting for $METRICS"
while [ ! -f "$METRICS" ]; do
  sleep 60
done
# settle: ensure the writer is done (mtime stable for 60s)
prev=$(stat -c %Y "$METRICS" 2>/dev/null || stat -f %m "$METRICS")
sleep 60
now=$(stat -c %Y "$METRICS" 2>/dev/null || stat -f %m "$METRICS")
while [ "$prev" != "$now" ]; do
  prev=$now; sleep 60
  now=$(stat -c %Y "$METRICS" 2>/dev/null || stat -f %m "$METRICS")
done
echo "[v8-check-watcher] $(date '+%F %T') metrics.json settled; running verification"
python scripts/v8_check.py
rc=$?
echo "[v8-check-watcher] $(date '+%F %T') verification exited rc=$rc"
exit $rc
