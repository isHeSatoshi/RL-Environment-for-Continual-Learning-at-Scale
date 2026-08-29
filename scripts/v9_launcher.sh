#!/usr/bin/env bash
# Launch the SCCL v9 (Branch G, G2 dose ladder) experiment in two stages:
#   stage 1: wiring smoke (configs/_smoke_v9.json, 2 families) — must exit 0
#            and pass the fail-closed engagement audit (G=2: first-contact
#            manufacture of TWO ':g' witnesses per family where certification
#            succeeds, ':g' ids in gate checked_cap_ids with cap_gen_pool=2,
#            theta dose fields {0.5, 2/3}, retirement/anti-contamination walk
#            clean incl. LIST-valued marker sources, holdout disjointness,
#            control isolation, anchor engagement on accepted updates);
#   stage 2: THE EXPERIMENT — all three seeds (42,43,44) sequentially via
#            scripts/run_seeds.py (the pre-registered v9 protocol: the 3-seed
#            ladder IS the experiment; no interim decision, no seed-42-only
#            gate), then scripts/v9_check.py for the fail-closed C1-C5 checks
#            and the pre-registered H1/H2/H3 + V9 VERDICT rules.
# Launch detached:
#   nohup bash scripts/v9_launcher.sh > runs/sccl_v9_launcher.log 2>&1 &
#
# DOUBLE-LAUNCH GUARD: refuse if another launcher holds the lock or if any
# v9 seed artifacts already exist.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/d/gcl_tmp
export TEMP=/d/gcl_tmp
export TMP=/d/gcl_tmp
mkdir -p /d/gcl_tmp

LOCK=runs/.v9_launcher.lock
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && tasklist //FI "PID eq $oldpid" 2>/dev/null | grep -qi bash; then
    echo "[v9] ABORT: launcher PID $oldpid still alive (lock $LOCK)"
    exit 1
  fi
  echo "[v9] stale lock (PID $oldpid dead); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for f in runs/sccl_v9_s42/metrics.json runs/sccl_v9_s43/metrics.json \
         runs/sccl_v9_s44/metrics.json runs/sccl_v9_seeds/aggregate.json; do
  if [ -f "$f" ]; then
    echo "[v9] ABORT: v9 artifacts already exist ($f)"
    exit 1
  fi
done
for f in runs/sccl_v8/metrics.json runs/sccl_v8_s43/metrics.json \
         runs/sccl_v8_s44/metrics.json; do
  if [ ! -f "$f" ]; then
    echo "[v9] ABORT: $f missing (C1 determinism reference)"
    exit 1
  fi
done

echo "[v9] $(date '+%F %T') settling 120s for GPU release"
sleep 120

echo "[v9] $(date '+%F %T') stage 1: wiring smoke (configs/_smoke_v9.json)"
python -m gcl.runner --config configs/_smoke_v9.json
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v9] ABORT: smoke run exited rc=$rc"
  exit $rc
fi
python - << 'PY'
import json, os, sys

# Fail-closed engagement audit for the v9 smoke (G=2 dose rows). Reads the
# same artifacts as scripts/v9_check.py so the smoke audit and the final
# checker agree by construction.
RUN = "runs/_smoke_v9"
CAP_N, CAP_MARGIN, G_LANE = 3, 2, 2
GEN_ROWS = ["sccl_gen2_half", "sccl_gen2_majority"]
THETA_ROWS = {"sccl_gen2_half": 0.5, "sccl_gen2_majority": 2 / 3}
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
    ids = set()
    m = os.path.join(RUN, "metrics.json")
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

# ---- control row: pre-v8/v9 gate shape, zero G-lane activity ----
recs = load_records(CONTROL)
g1_leak = [g for g in gates_of(recs)
           if ("cap_gen_pool" in g) or ("checked_cap_ids" in g)]
gen_markers = [r for r in recs if r.get("sccl_gen_probe")]
cg = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
if g1_leak or gen_markers:
    fails.append(f"control isolation: {CONTROL} shows gen activity "
                 f"(fields={len(g1_leak)}, markers={len(gen_markers)})")
elif not cg:
    fails.append(f"control: {CONTROL} never checked cap probes "
                 "(cap stratum not engaged)")
else:
    notes.append(f"control OK: {CONTROL} checked-cap gates={len(cg)}, "
                 "no G fields, no markers")

# ---- treatment rows: G=2 manufacture, lanes in gates, retirement walk ----
for name in GEN_ROWS:
    recs = load_records(name)
    if not recs:
        fails.append(f"G2: {name} has no trajectory records")
        continue
    markers = [r for r in recs if r.get("sccl_gen_probe")]
    lane_sizes = [len(m["sccl_gen_probe"].get("src") or [])
                 for m in markers]
    gates = gates_of(recs)
    gen_gates = [g for g in gates
                 if any(i.endswith(":g") for i in g.get("checked_cap_ids", []))]
    if not markers:
        fails.append(f"G2: {name} manufactured no gen probe at first contact")
    if lane_sizes and max(lane_sizes) < 1:
        fails.append(f"G2: {name} first-contact markers carry empty src lanes")
    if not gen_gates:
        fails.append(f"G2: {name} never checked a ':g' probe at any gate")
    trained = set()
    bad_retire, bad_holdout = [], []
    for r in recs:
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
        for msrc in (r.get("sccl_gen_probe") or {}).get("src", []) or []:
            if msrc in HOLD:
                bad_holdout.append(msrc)
            if msrc in trained:
                bad_retire.append((r.get("task_id"), msrc + ":g (mfd)"))
    if bad_retire:
        fails.append(f"G2 retirement: {name} checked a gen probe whose source "
                    f"was already trained: {bad_retire[:3]}")
    if bad_holdout:
        fails.append(f"G2 contamination: {name} gen probes sourced from GOLD "
                     f"HOLDOUT tasks: {sorted(set(bad_holdout))}")
    if markers and gen_gates and not (bad_retire or bad_holdout):
        notes.append(f"G2 OK: {name} markers={len(markers)} first-contact "
                     f"lane sizes={lane_sizes} gen-checked gates="
                     f"{len(gen_gates)} retirement walk clean")

    # dose/shape: theta, cap_n, gen_pool, denominators, ':g' in cap_rates
    cg2 = [g for g in gates if int(g.get("checked_cap", 0)) > 0]
    theta = THETA_ROWS[name]
    holes = [g for g in cg2 if not g.get("cap_rates")]
    bad_theta = [g for g in cg2
                 if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
    bad_n = [g for g in cg2 if int(g.get("cap_n", -1)) != CAP_N]
    bad_pool = [g for g in cg2 if int(g.get("cap_gen_pool", -1)) != G_LANE]
    denoms = sorted({int(x["n"]) for g in cg2
                     for x in (g.get("cap_rates") or {}).values()})
    bad_denoms = [d for d in denoms if d not in (CAP_N, CAP_N + CAP_MARGIN)]
    gen_rates = [tid for g in cg2 for tid in (g.get("cap_rates") or {})
                 if tid.endswith(":g")]
    if holes:
        fails.append(f"dose: {name} has {len(holes)} checked-cap gates with "
                     "EMPTY cap_rates")
    if bad_theta:
        fails.append(f"dose: {name} cap_retain_min != {theta} on "
                     f"{len(bad_theta)} gates")
    if bad_n:
        fails.append(f"dose: {name} cap_n != {CAP_N} on {len(bad_n)} gates")
    if bad_pool:
        fails.append(f"dose: {name} cap_gen_pool != {G_LANE} on "
                     f"{len(bad_pool)} gates")
    if bad_denoms:
        fails.append(f"dose: {name} rate denominators {bad_denoms} outside "
                     f"{{3, {CAP_N + CAP_MARGIN}}}")
    if gen_gates and not gen_rates:
        fails.append(f"dose: {name} checked ':g' probes but cap_rates never "
                     "covered a ':g' id")
    if not (holes or bad_theta or bad_n or bad_pool or bad_denoms):
        notes.append(f"dose OK: {name} checked-cap gates={len(cg2)} "
                     f"theta={theta} denoms={denoms} gen-rate ids="
                     f"{len(gen_rates)}")

    # anchor engagement (both treatment rows carry the anchor)
    acc = accepted_infos(recs)
    anch = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
    lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
    if acc and len(anch) != len(acc):
        fails.append(f"anchor: {name} {len(acc) - len(anch)}/{len(acc)} "
                     "accepted updates with anchor_pen == 0")
    elif acc and lams != {0.1}:
        fails.append(f"anchor: {name} lambdas {sorted(lams)} != [0.1]")
    elif acc:
        notes.append(f"anchor OK: {name} {len(anch)}/{len(acc)} accepted "
                     "updates anchor_pen>0 lambda=0.1")
    else:
        notes.append(f"anchor note: {name} 0 accepted updates in smoke "
                     "(wiring only; cannot verify anchor engagement here)")

for n in notes:
    print(f"[v9-smoke-audit] {n}")
if fails:
    for f in fails:
        print(f"[v9-smoke-audit] FAIL: {f}")
    print(f"[v9-smoke-audit] -> FAIL ({len(fails)} failures)")
    sys.exit(1)
print("[v9-smoke-audit] -> PASS")
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "[v9] ABORT: smoke audit failed — G2 lane or dose wiring broken"
  exit $rc
fi

echo "[v9] $(date '+%F %T') stage 2: the experiment — seeds 42,43,44 (sequential, ~13.5h)"
python scripts/run_seeds.py --config configs/sccl_v9.json --seeds 42,43,44
rc=$?
echo "[v9] $(date '+%F %T') seed runs exited rc=$rc"
if [ $rc -ne 0 ]; then
  exit $rc
fi

echo "[v9] $(date '+%F %T') running the pre-registered verification (v9_check.py)"
python scripts/v9_check.py
rc=$?
echo "[v9] $(date '+%F %T') verification exited rc=$rc"
exit $rc
