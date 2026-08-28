#!/usr/bin/env bash
# Launch the SCCL v7 (Branch E, E1 pass-rate margin veto) ladder AFTER the v6
# run is final, in two stages:
#   stage 1: wiring smoke (configs/_smoke_v7.json, 2 families x 4 tasks) — must
#            exit 0 and pass the fail-closed engagement audit (E1 cap_rates
#            engaged on theta>0 rows with correct dose, theta=0 row keeps the
#            legacy gate shape, anchor engaged on the composition row, sccl
#            control isolated); abort the ladder otherwise;
#   stage 2: the full ladder (configs/sccl_v7.json -> runs/sccl_v7).
# Launch detached:
#   nohup bash scripts/v7_launcher.sh > runs/sccl_v7_launcher.log 2>&1 &
#
# DOUBLE-LAUNCH GUARD (post-mortem in RESEARCH_NOTES_v5.md): refuse to start if
# another copy holds the lock, or if v7 artifacts already exist.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v7_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v7] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v7] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -f runs/sccl_v7/metrics.json ] || [ -f runs/sccl_v7/metrics_partial.json ]; then
  echo "[v7] ABORT: runs/sccl_v7 artifacts already exist"
  exit 1
fi

echo "[v7] $(date '+%F %T') waiting for runs/sccl_v6/metrics.json (determinism reference) ..."
while [ ! -f runs/sccl_v6/metrics.json ]; do
  sleep 120
done

echo "[v7] $(date '+%F %T') settling 120s for GPU release"
sleep 120

echo "[v7] $(date '+%F %T') stage 1: wiring smoke (configs/_smoke_v7.json)"
python -m gcl.runner --config configs/_smoke_v7.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v7] ABORT: smoke run exited rc=$rc"
  exit $rc
fi
python - << 'PY'
import json, os, sys

# Fail-closed engagement audit for the v7 smoke. Reads the same artifacts as
# scripts/v7_check.py (trajectories_<name>.jsonl update_info.gate, vault
# skills.json, metrics.json) so the smoke audit and the final verdict checker
# agree by construction.
RUN = "runs/_smoke_v7"
CAP_N, CAP_MARGIN = 3, 2

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

fails, notes = [], []

# ---- C4: sccl control isolation (no cap probes, no cap stratum, no E1) ----
recs = load_updates("sccl")
checked = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
e1_leak = [g for g in gates_of(recs)
           if ("cap_rates" in g) or ("cap_retain_min" in g) or ("cap_n" in g)]
if checked or e1_leak:
    fails.append(f"C4 isolation: sccl leaked (checked_cap_gates={len(checked)}, "
                 f"e1_fields={len(e1_leak)})")
else:
    notes.append("C4 isolation OK: sccl checked_cap=0, no E1 fields")

# ---- theta=0 cap row: stratum engaged, legacy gate shape (no E1 fields) ----
recs = load_updates("sccl_capprobe_strat")
cg = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
e1_leak = [g for g in gates_of(recs)
           if ("cap_rates" in g) or ("cap_retain_min" in g) or ("cap_n" in g)]
if not cg:
    fails.append("legacy: sccl_capprobe_strat never checked cap probes "
                 "(cap stratum not engaged)")
elif e1_leak:
    fails.append(f"legacy: sccl_capprobe_strat (theta=0) logged E1 fields on "
                 f"{len(e1_leak)} gates (bit-identity shape violated)")
else:
    notes.append(f"legacy shape OK: sccl_capprobe_strat checked-cap gates={len(cg)}, "
                 "no E1 fields")

# ---- E1 rows: cap_rates engaged with the configured dose ----
for name, theta in [("sccl_strict", 1.0), ("sccl_ens_strict_anchor", 1.0)]:
    recs = load_updates(name)
    cg = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
    if not cg:
        fails.append(f"E1: {name} never checked cap probes")
        continue
    holes = [g for g in cg if not g.get("cap_rates")]
    bad_theta = [g for g in cg
                 if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
    bad_n = [g for g in cg if int(g.get("cap_n", -1)) != CAP_N]
    denoms = sorted({int(r["n"]) for g in cg
                     for r in (g.get("cap_rates") or {}).values()})
    bad_denoms = [d for d in denoms if d not in (CAP_N, CAP_N + CAP_MARGIN)]
    if holes:
        fails.append(f"E1: {name} has {len(holes)} checked-cap gates with EMPTY "
                     "cap_rates")
    if bad_theta:
        fails.append(f"E1: {name} logged cap_retain_min != {theta} on "
                     f"{len(bad_theta)} gates")
    if bad_n:
        fails.append(f"E1: {name} logged cap_n != {CAP_N} on {len(bad_n)} gates")
    if bad_denoms:
        fails.append(f"E1: {name} rate denominators {bad_denoms} outside "
                     f"{{3, {CAP_N + CAP_MARGIN}}}")
    if not (holes or bad_theta or bad_n or bad_denoms):
        n_veto = sum(1 for g in cg if g.get("broke_cap"))
        notes.append(f"E1 OK: {name} checked-cap gates={len(cg)} "
                     f"cap-vetoes={n_veto} denoms={denoms} theta={theta}")

# ---- anchor engagement on the composition row ----
recs = load_updates("sccl_ens_strict_anchor")
acc = accepted_infos(recs)
anch = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
if acc and len(anch) != len(acc):
    fails.append(f"anchor: sccl_ens_strict_anchor {len(acc) - len(anch)}/{len(acc)} "
                 "accepted updates with anchor_pen == 0")
elif acc and lams != {0.1}:
    fails.append(f"anchor: sccl_ens_strict_anchor lambdas {sorted(lams)} != [0.1]")
elif acc:
    notes.append(f"anchor OK: sccl_ens_strict_anchor {len(anch)}/{len(acc)} accepted "
                 "updates anchor_pen>0 lambda=0.1")
else:
    notes.append("anchor note: sccl_ens_strict_anchor had 0 accepted updates "
                 "(cannot verify anchor)")

# ---- ensemble pool structure on the composition row (K=3) ----
caps = cap_probes_in_vault("sccl_ens_strict_anchor")
per_fam = {}
for c in caps:
    per_fam.setdefault(c["family"], []).append(src(c["task_id"]))
probs = []
for f, srcs in per_fam.items():
    if len(srcs) > 3:
        probs.append(f"{f}: {len(srcs)} probes > K=3")
    if len(srcs) != len(set(srcs)):
        probs.append(f"{f}: duplicate source skill not deduped")
if probs:
    fails.append(f"ensemble pool structure broken: {probs}")
else:
    distinct = {f: len(set(s)) for f, s in per_fam.items()}
    notes.append(f"ensemble pool OK: distinct-per-family={distinct}")

for n in notes:
    print(f"[v7-smoke-audit] {n}")
if fails:
    for f in fails:
        print(f"[v7-smoke-audit] FAIL: {f}")
    print(f"[v7-smoke-audit] -> FAIL ({len(fails)} failures)")
    sys.exit(1)
print("[v7-smoke-audit] -> PASS")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v7] ABORT: smoke audit failed — E1 not engaged, dose wrong, or control leaked"
  exit $rc
fi

echo "[v7] $(date '+%F %T') stage 2: full ladder (configs/sccl_v7.json)"
python -m gcl.runner --config configs/sccl_v7.json
rc=$?
echo "[v7] $(date '+%F %T') ladder exited rc=$rc"
exit $rc
