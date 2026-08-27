#!/usr/bin/env bash
# Wait for the v6 (Branch D) capability-ensemble + anchor ladder (runs/sccl_v6)
# to finalize, then run the pre-registered verification (fail-closed C1-C5 +
# H1/H2/H3/BREAKTHROUGH). metrics.json is written only at run completion;
# during the run only metrics_partial.json exists.
set -u
cd "$(dirname "$0")/.."
METRICS=runs/sccl_v6/metrics.json
echo "[v6-check-watcher] $(date '+%F %T') waiting for $METRICS"
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
echo "[v6-check-watcher] $(date '+%F %T') metrics.json settled; running verification"
python scripts/v6_check.py
rc=$?
echo "[v6-check-watcher] $(date '+%F %T') verification exited rc=$rc"
exit $rc
