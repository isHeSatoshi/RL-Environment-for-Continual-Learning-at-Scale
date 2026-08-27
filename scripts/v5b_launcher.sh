#!/usr/bin/env bash
# Launch the SCCL v5b capability-probe ladder AFTER the corrected v5 factorial
# is final, in two stages:
#   stage 1: wiring smoke (configs/_smoke_v5b.json, 2 families x 2 tasks) —
#            must exit 0, commit cap probes, and show the cap stratum checking
#            them in gate records; abort the ladder otherwise;
#   stage 2: the full ladder (configs/sccl_v5b.json -> runs/sccl_v5b).
# Launch detached:
#   nohup bash scripts/v5b_launcher.sh > runs/sccl_v5b_launcher.log 2>&1 &
#
# DOUBLE-LAUNCH GUARD (post-mortem in RESEARCH_NOTES_v5.md): refuse to start
# if another copy holds the lock, if a GCL runner is already flying
# (runs/sccl_v5_fixed in progress), or if v5b artifacts already exist.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v5b_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v5b] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v5b] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -f runs/sccl_v5b/metrics.json ] || [ -f runs/sccl_v5b/metrics_partial.json ]; then
  echo "[v5b] ABORT: runs/sccl_v5b artifacts already exist"
  exit 1
fi

echo "[v5b] $(date '+%F %T') waiting for runs/sccl_v5_fixed/metrics.json (corrected factorial final) ..."
while [ ! -f runs/sccl_v5_fixed/metrics.json ]; do
  sleep 120
done

echo "[v5b] $(date '+%F %T') settling 120s for GPU release after the v5 run"
sleep 120

echo "[v5b] $(date '+%F %T') stage 1: wiring smoke (configs/_smoke_v5b.json)"
python -m gcl.runner --config configs/_smoke_v5b.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v5b] ABORT: smoke run exited rc=$rc"
  exit $rc
fi
python - << 'PY'
import json, os, sys
m = json.load(open("runs/_smoke_v5b/metrics.json"))["learners"]
cap = m.get("sccl_capprobe_strat", {})
st = cap.get("sccl", {}) or {}
committed = int(st.get("cap_probes_committed", 0))
gates = [((e.get("update_info") or {}).get("gate") or {})
         for e in cap.get("trajectories", [])]
checked = [g for g in gates if int(g.get("checked_cap", 0)) > 0]
sccl = m.get("sccl", {})
sst = sccl.get("sccl", {}) or {}
leak = int(sst.get("cap_probes_committed", 0))
ok = committed > 0 and len(checked) > 0 and leak == 0
print(f"[v5b-smoke-audit] cap_probes_committed={committed} "
      f"gates_with_checked_cap>0={len(checked)} sccl_control_leak={leak} -> "
      f"{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v5b] ABORT: smoke audit failed — cap probes not engaged or control leaked"
  exit $rc
fi

echo "[v5b] $(date '+%F %T') stage 2: full ladder (configs/sccl_v5b.json)"
python -m gcl.runner --config configs/sccl_v5b.json
rc=$?
echo "[v5b] $(date '+%F %T') ladder exited rc=$rc"
exit $rc
