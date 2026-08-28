#!/usr/bin/env bash
# Launch the SCCL v8 (Branch F, G1 generalization witnesses) ladder in two
# stages:
#   stage 1: wiring smoke (configs/_smoke_v8.json, 2 families x 4 tasks) — must
#            exit 0 and pass the fail-closed engagement audit (gen probes
#            manufactured at first contact, ':g' ids in gate checked_cap_ids,
#            retirement/anti-contamination walk clean, theta rows log cap_rates
#            with correct dose, theta=0 gen row keeps the legacy gate shape,
#            control row shows zero G1 activity, anchor engaged); abort the
#            ladder otherwise;
#   stage 2: the full ladder (configs/sccl_v8.json -> runs/sccl_v8).
# Launch detached:
#   nohup bash scripts/v8_launcher.sh > runs/sccl_v8_launcher.log 2>&1 &
#
# DOUBLE-LAUNCH GUARD (post-mortem in RESEARCH_NOTES_v5.md): refuse to start if
# another copy holds the lock, or if v8 artifacts already exist.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v8_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v8] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v8] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -f runs/sccl_v8/metrics.json ] || [ -f runs/sccl_v8/metrics_partial.json ]; then
  echo "[v8] ABORT: runs/sccl_v8 artifacts already exist"
  exit 1
fi

if [ ! -f runs/sccl_v7/metrics.json ]; then
  echo "[v8] ABORT: runs/sccl_v7/metrics.json missing (C1 determinism reference)"
  exit 1
fi

echo "[v8] $(date '+%F %T') settling 120s for GPU release"
sleep 120

echo "[v8] $(date '+%F %T') stage 1: wiring smoke (configs/_smoke_v8.json)"
python -m gcl.runner --config configs/_smoke_v8.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v8] ABORT: smoke run exited rc=$rc"
  exit $rc
fi
python - << 'PY'
import json, os, sys

# Fail-closed engagement audit for the v8 smoke. Reads the same artifacts as
# scripts/v8_check.py (trajectories_<name>.jsonl update_info.gate +
# sccl_gen_probe markers, vault skills.json, metrics.json) so the smoke audit
# and the final verdict checker agree by construction.
RUN = "runs/_smoke_v8"
CAP_N, CAP_MARGIN = 3, 2
GEN_ROWS = ["sccl_genprobe", "sccl_genprobe_strict", "sccl_genprobe_strict_anchor"]
THETA_ROWS = {"sccl_genprobe_strict": 1.0, "sccl_genprobe_strict_anchor": 1.0}
CONTROL = "sccl_capprobe_strat"

def load_records(name):
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

def holdout_ids():
    m = os.path.join(RUN, "metrics.json")
    ids = set()
    if os.path.exists(m):
        for L in json.load(open(m)).get("learners", {}).values():
            for fam, recs in (L.get("eval_detail", {})
                              .get("final_heldout", {}) or {}).items():
                for t in recs or []:
                    if t.get("task_id"):
                        ids.add(t["task_id"])
    return ids

fails, notes = [], []
HOLD = holdout_ids()

# ---- control row: pre-v8 gate shape, zero G1 activity ----
recs = load_records(CONTROL)
g1_leak = [g for g in gates_of(recs)
           if ("cap_gen_pool" in g) or ("checked_cap_ids" in g)]
gen_markers = [r for r in recs if r.get("sccl_gen_probe")]
cg = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
vp = os.path.join(RUN, f"vault_{CONTROL}", "skills.json")
vg = [r for r in (json.load(open(vp)) if os.path.exists(vp) else [])
      if r.get("kind") == "cap_probe" and r["task_id"].endswith(":g")]
if g1_leak or gen_markers or vg:
    fails.append(f"control isolation: {CONTROL} shows G1 activity "
                 f"(g1_fields={len(g1_leak)}, markers={len(gen_markers)}, "
                 f"gen_probes_in_vault={len(vg)})")
elif not cg:
    fails.append(f"control: {CONTROL} never checked cap probes "
                 "(cap stratum not engaged)")
else:
    notes.append(f"control OK: {CONTROL} checked-cap gates={len(cg)}, "
                 "no G1 fields, no ':g' probes")

# ---- gen rows: manufacture at first contact + ':g' in gates + retirement ----
for name in GEN_ROWS:
    recs = load_records(name)
    if not recs:
        fails.append(f"G1: {name} has no trajectory records")
        continue
    markers = [r for r in recs if r.get("sccl_gen_probe")]
    gates = gates_of(recs)
    gen_gates = [g for g in gates
                 if any(i.endswith(":g") for i in g.get("checked_cap_ids", []))]
    if not markers:
        fails.append(f"G1: {name} manufactured no gen probe at first contact "
                     "(no sccl_gen_probe marker)")
    if not gen_gates:
        fails.append(f"G1: {name} never checked a ':g' probe at any gate")
    # retirement/anti-contamination walk: no ':g' from task T checked during
    # or after T's own episode; no holdout id anywhere
    trained = set()
    bad_retire, bad_holdout = [], []
    for r in recs:
        # count the record's own task as in-training BEFORE inspecting its gate
        if r.get("task_id"):
            trained.add(r["task_id"])
        g = (r.get("update_info") or {}).get("gate") or {}
        for i in g.get("checked_cap_ids", []):
            if not i.endswith(":g"):
                continue
            src = i[:-2]
            if src in trained:
                bad_retire.append((r.get("task_id"), i))
            if src in HOLD:
                bad_holdout.append(i)
        msrc = (r.get("sccl_gen_probe") or {}).get("src")
        if msrc and msrc in HOLD:
            bad_holdout.append(msrc)
    if bad_retire:
        fails.append(f"G1 retirement: {name} checked a gen probe whose source "
                     f"was already trained: {bad_retire[:3]}")
    if bad_holdout:
        fails.append(f"G1 contamination: {name} gen probes sourced from GOLD "
                     f"HOLDOUT tasks: {sorted(set(bad_holdout))}")
    if markers and gen_gates and not (bad_retire or bad_holdout):
        notes.append(f"G1 OK: {name} first-contact markers={len(markers)} "
                     f"gen-checked gates={len(gen_gates)} retirement walk clean")

    # dose/shape per row type
    cg = [g for g in gates if int(g.get("checked_cap", 0)) > 0]
    if name in THETA_ROWS:
        theta = THETA_ROWS[name]
        holes = [g for g in cg if not g.get("cap_rates")]
        bad_theta = [g for g in cg
                     if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
        bad_n = [g for g in cg if int(g.get("cap_n", -1)) != CAP_N]
        denoms = sorted({int(x["n"]) for g in cg
                         for x in (g.get("cap_rates") or {}).values()})
        bad_denoms = [d for d in denoms if d not in (CAP_N, CAP_N + CAP_MARGIN)]
        gen_rates = [tid for g in cg for tid in (g.get("cap_rates") or {})
                     if tid.endswith(":g")]
        if holes:
            fails.append(f"E1: {name} has {len(holes)} checked-cap gates with "
                         "EMPTY cap_rates")
        if bad_theta:
            fails.append(f"E1: {name} logged cap_retain_min != {theta} on "
                         f"{len(bad_theta)} gates")
        if bad_n:
            fails.append(f"E1: {name} logged cap_n != {CAP_N} on {len(bad_n)} gates")
        if bad_denoms:
            fails.append(f"E1: {name} rate denominators {bad_denoms} outside "
                         f"{{3, {CAP_N + CAP_MARGIN}}}")
        if gen_gates and not gen_rates:
            fails.append(f"E1: {name} checked ':g' probes but cap_rates never "
                         "covered a ':g' id")
        if not (holes or bad_theta or bad_n or bad_denoms):
            notes.append(f"E1 OK: {name} checked-cap gates={len(cg)} "
                         f"cap-vetoes={sum(1 for g in cg if g.get('broke_cap'))} "
                         f"denoms={denoms} gen-rate ids={len(gen_rates)}")
    else:
        e1_leak = [g for g in gates if ("cap_rates" in g)
                   or ("cap_retain_min" in g) or ("cap_n" in g)]
        if e1_leak:
            fails.append(f"legacy: {name} (theta=0) logged E1 fields on "
                         f"{len(e1_leak)} gates (bit-identity shape violated)")
        elif cg:
            notes.append(f"legacy shape OK: {name} checked-cap gates={len(cg)}, "
                         "no E1 fields")

# ---- anchor engagement on the composition row ----
recs = load_records("sccl_genprobe_strict_anchor")
acc = accepted_infos(recs)
anch = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
if acc and len(anch) != len(acc):
    fails.append(f"anchor: sccl_genprobe_strict_anchor "
                 f"{len(acc) - len(anch)}/{len(acc)} accepted updates with "
                 "anchor_pen == 0")
elif acc and lams != {0.1}:
    fails.append(f"anchor: sccl_genprobe_strict_anchor lambdas {sorted(lams)} "
                 "!= [0.1]")
elif acc:
    notes.append(f"anchor OK: {len(anch)}/{len(acc)} accepted updates "
                 "anchor_pen>0 lambda=0.1")
else:
    notes.append("anchor note: 0 accepted updates on the anchor row "
                 "(cannot verify anchor in smoke)")

for n in notes:
    print(f"[v8-smoke-audit] {n}")
if fails:
    for f in fails:
        print(f"[v8-smoke-audit] FAIL: {f}")
    print(f"[v8-smoke-audit] -> FAIL ({len(fails)} failures)")
    sys.exit(1)
print("[v8-smoke-audit] -> PASS")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v8] ABORT: smoke audit failed — G1 not engaged, retirement broken, or control leaked"
  exit $rc
fi

echo "[v8] $(date '+%F %T') stage 2: full ladder (configs/sccl_v8.json)"
python -m gcl.runner --config configs/sccl_v8.json
rc=$?
echo "[v8] $(date '+%F %T') ladder exited rc=$rc"
exit $rc
