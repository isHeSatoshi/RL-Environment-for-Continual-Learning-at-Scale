#!/usr/bin/env bash
# Launch the SCCL v6 (Branch D) capability-ENSEMBLE + anchor ladder AFTER the
# v5b capability-probe run is final, in two stages:
#   stage 1: wiring smoke (configs/_smoke_v6.json, 2 families x 4 tasks) — must
#            exit 0 and pass the fail-closed engagement audit (ensemble pool
#            engaged + structurally correct, anchor engaged on anchor rows and
#            absent elsewhere, sccl control isolated); abort the ladder otherwise;
#   stage 2: the full ladder (configs/sccl_v6.json -> runs/sccl_v6).
# Launch detached:
#   nohup bash scripts/v6_launcher.sh > runs/sccl_v6_launcher.log 2>&1 &
#
# DOUBLE-LAUNCH GUARD (post-mortem in RESEARCH_NOTES_v5.md): refuse to start if
# another copy holds the lock, if a GCL runner is already flying, or if v6
# artifacts already exist.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v6_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v6] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v6] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -f runs/sccl_v6/metrics.json ] || [ -f runs/sccl_v6/metrics_partial.json ]; then
  echo "[v6] ABORT: runs/sccl_v6 artifacts already exist"
  exit 1
fi

echo "[v6] $(date '+%F %T') waiting for runs/sccl_v5b/metrics.json (determinism reference) ..."
while [ ! -f runs/sccl_v5b/metrics.json ]; do
  sleep 120
done

echo "[v6] $(date '+%F %T') settling 120s for GPU release"
sleep 120

echo "[v6] $(date '+%F %T') stage 1: wiring smoke (configs/_smoke_v6.json)"
python -m gcl.runner --config configs/_smoke_v6.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v6] ABORT: smoke run exited rc=$rc"
  exit $rc
fi
python - << 'PY'
import json, os, sys

# Fail-closed engagement audit for the v6 smoke. Per-step gate records live in
# trajectories_<name>.jsonl (update_info.gate) and the anchor penalty in
# update_info.anchor_pen; the cap-probe pool is read from vault_<name>/skills.json.
# This mirrors the evidence readers in scripts/v6_check.py so the smoke audit and
# the final verdict checker read identical artifacts.
RUN = "runs/_smoke_v6"

def load_updates(name):
    p = os.path.join(RUN, f"trajectories_{name}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def update_infos(recs):
    return [e["update_info"] for e in recs if e.get("update_info")]

def gates_of(recs):
    return [g for g in (ui.get("gate") for ui in update_infos(recs)) if g]

def accepted_infos(recs):
    return [ui for ui in update_infos(recs) if ui.get("accepted")]

def cap_probes_in_vault(name):
    p = os.path.join(RUN, f"vault_{name}", "skills.json")
    if not os.path.exists(p):
        return []
    return [r for r in json.load(open(p)) if r.get("kind") == "cap_probe"]

def src(task_id):
    return task_id.rsplit(":c", 1)[0]

def pool_structure(name, K):
    """Return (distinct_src_per_family, structural_problems). The ensemble pool
    must retain the freshest variant of up to K DISTINCT source skills per
    family: never >K probes in a family, never two probes sharing a source."""
    per_fam = {}
    for c in cap_probes_in_vault(name):
        per_fam.setdefault(c["family"], []).append(src(c["task_id"]))
    distinct = {f: len(set(s)) for f, s in per_fam.items()}
    probs = []
    for f, srcs in per_fam.items():
        if len(srcs) > K:
            probs.append(f"{f}: {len(srcs)} probes > K={K}")
        if len(srcs) != len(set(srcs)):
            probs.append(f"{f}: duplicate source skill not deduped to freshest")
    return distinct, probs

m = json.load(open(os.path.join(RUN, "metrics.json")))["learners"]
fails, notes = [], []

# ---- C4: sccl control isolation (no cap probes, no cap stratum, no anchor) ----
st = (m.get("sccl", {}).get("sccl") or {})
recs = load_updates("sccl")
leak_committed = int(st.get("cap_probes_committed", 0))
leak_checked = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
leak_anchor = [ui for ui in accepted_infos(recs)
               if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
if leak_committed or leak_checked or leak_anchor:
    fails.append(f"C4 isolation: sccl leaked (cap_committed={leak_committed}, "
                 f"checked_cap_gates={len(leak_checked)}, anchor_updates={len(leak_anchor)})")
else:
    notes.append("C4 isolation OK: sccl cap_committed=0 checked_cap=0 anchor_pen=0")

# ---- C2 (ensemble) + C3-off: sccl_capens (K=3, anchor OFF) ----
st = (m.get("sccl_capens", {}).get("sccl") or {})
committed = int(st.get("cap_probes_committed", 0))
distinct, probs = pool_structure("sccl_capens", 3)
recs = load_updates("sccl_capens")
max_checked = max([int(g.get("checked_cap", 0)) for g in gates_of(recs)], default=0)
ens_anchor = [ui for ui in accepted_infos(recs)
              if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
if committed == 0:
    fails.append("C2: sccl_capens committed no cap probes (manufacture not engaged)")
if probs:
    fails.append(f"C2: sccl_capens pool structure broken: {probs}")
if ens_anchor:
    fails.append(f"C3-off: sccl_capens logged anchor_pen>0 on {len(ens_anchor)} "
                 "accepted updates (anchor leaked onto a non-anchor row)")
max_distinct = max(distinct.values(), default=0)
if max_distinct < 2:
    fails.append(f"C2: sccl_capens never held >=2 distinct-source-skill probes in a "
                 f"family (distinct-per-family={distinct}); the ensemble path was not "
                 "exercised by the smoke — increase smoke tasks before the ladder")
elif max_checked < 2:
    fails.append(f"C2: sccl_capens held {max_distinct} distinct probes in a family but "
                 f"max checked_cap={max_checked} < 2 (ensemble veto stratum not engaged)")
else:
    notes.append(f"C2 ensemble OK: sccl_capens distinct-per-family={distinct} "
                 f"max_checked_cap={max_checked} anchor_pen=0")

# ---- C3 (anchor) + K=1: sccl_cap_anchor (K=1, anchor ON) ----
recs = load_updates("sccl_cap_anchor")
acc = accepted_infos(recs)
anch = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
distinct1, probs1 = pool_structure("sccl_cap_anchor", 1)
if probs1:
    fails.append(f"C2/K1: sccl_cap_anchor pool structure broken: {probs1}")
if acc and not anch:
    fails.append(f"C3: sccl_cap_anchor had {len(acc)} accepted updates but none logged "
                 "anchor_pen>0 (anchor not engaged)")
elif acc and anch and lams != {0.1}:
    fails.append(f"C3: sccl_cap_anchor anchor_lambda {sorted(lams)} != [0.1]")
elif acc:
    notes.append(f"C3 anchor OK: sccl_cap_anchor {len(anch)}/{len(acc)} accepted updates "
                 "anchor_pen>0 lambda=0.1")
else:
    notes.append("C3 note: sccl_cap_anchor had 0 accepted updates (cannot verify anchor)")

# ---- C2 + C3 composition: sccl_capens_anchor (K=3, anchor ON) ----
st = (m.get("sccl_capens_anchor", {}).get("sccl") or {})
committed = int(st.get("cap_probes_committed", 0))
distinct, probs = pool_structure("sccl_capens_anchor", 3)
recs = load_updates("sccl_capens_anchor")
acc = accepted_infos(recs)
anch = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
max_checked = max([int(g.get("checked_cap", 0)) for g in gates_of(recs)], default=0)
if committed == 0:
    fails.append("C2: sccl_capens_anchor committed no cap probes")
if probs:
    fails.append(f"C2: sccl_capens_anchor pool structure broken: {probs}")
if acc and not anch:
    fails.append(f"C3: sccl_capens_anchor had {len(acc)} accepted updates but none "
                 "logged anchor_pen>0 (anchor not engaged)")
elif acc and anch and lams != {0.1}:
    fails.append(f"C3: sccl_capens_anchor anchor_lambda {sorted(lams)} != [0.1]")
max_distinct = max(distinct.values(), default=0)
if max_distinct >= 2 and max_checked < 2:
    fails.append(f"C2: sccl_capens_anchor held {max_distinct} distinct probes but "
                 f"max checked_cap={max_checked} < 2")
if committed and (acc or max_checked):
    notes.append(f"C2+C3 composition: sccl_capens_anchor distinct-per-family={distinct} "
                 f"max_checked_cap={max_checked} anchor_updates={len(anch)}/{len(acc)}")

for n in notes:
    print(f"[v6-smoke-audit] {n}")
if fails:
    for f in fails:
        print(f"[v6-smoke-audit] FAIL: {f}")
    print(f"[v6-smoke-audit] -> FAIL ({len(fails)} failures)")
    sys.exit(1)
print("[v6-smoke-audit] -> PASS")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v6] ABORT: smoke audit failed — ensemble/anchor not engaged or control leaked"
  exit $rc
fi

echo "[v6] $(date '+%F %T') stage 2: full ladder (configs/sccl_v6.json)"
python -m gcl.runner --config configs/sccl_v6.json
rc=$?
echo "[v6] $(date '+%F %T') ladder exited rc=$rc"
exit $rc
