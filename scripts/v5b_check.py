"""Post-run verification for SCCL v5b capability probes (runs/sccl_v5b).

Run AFTER runs/sccl_v5b/metrics.json is final:
    python scripts/v5b_check.py

Checks, in order:
  1. INERTNESS + DETERMINISM — the v5b switches default off, so the frozen and
     sccl rows (seeds 42/43, unchanged indices) must bit-reproduce the
     corrected v5 factorial (runs/sccl_v5_fixed) on every headline metric and
     per-family holdout. Any mismatch means the v5b code leaked into the
     shared path.
  2. CAP-PROBE TELEMETRY — the two capability learners must have manufactured
     and committed probes (sccl_stats.cap_probes_*), their gate entries must
     show the cap stratum actually re-checking probes (checked_cap > 0 on
     gates once the vault holds probes), and the plain-sccl control must show
     none of it (isolation). Guard arming events are reported, not judged:
     the bounded-damage guard is allowed to arm; its rate is telemetry.
  3. PRE-REGISTERED DECISION RULES (RESEARCH_NOTES_v5.md Branch C +
     configs/sccl_v5b.json comment), fixed before this run:
       H1  (cap main effect):  sccl_capprobe.arith > sccl.arith AND
                               sccl_capprobe.frontier >= sccl.frontier - 0.02
       H2  (stratified cap):   sccl_capprobe_strat.arith >= sccl_capprobe.arith
                               AND frontier >= sccl.frontier - 0.02
       BREAKTHROUGH:           sccl_capprobe_strat.arith >= 0.55 AND
                               frontier >= sccl.frontier - 0.02
  4. VERDICT TABLE written to runs/sccl_v5b/verdict_check.md.

Gold labels never enter any check here except as the evaluation metric
(holdout scores already recorded by the runner at measurement time).
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5 = os.path.join(REPO, "runs", "sccl_v5_fixed")
V5B = os.path.join(REPO, "runs", "sccl_v5b")
CONTROLS = ["frozen", "sccl"]
CAP = ["sccl_capprobe", "sccl_capprobe_strat"]
REPORT_KEYS = ["acc", "bwt", "fwt", "forgetting", "auc", "stability", "updates"]
FAMILIES = ["arith", "math_word", "string", "drift"]


def load_metrics(run_dir: str) -> dict:
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        print(f"[check] {p} missing — run not final; aborting")
        sys.exit(1)
    return json.load(open(p))["learners"]


def fam_scores(learner: dict) -> dict:
    fh = learner.get("eval_detail", {}).get("final_heldout", {})
    out = {}
    for fam in FAMILIES:
        tasks = fh.get(fam, [])
        out[fam] = sum(t["score"] for t in tasks) / max(1, len(tasks))
    return out


def load_updates(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, f"trajectories_{name}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    v5 = load_metrics(V5)
    v5b = load_metrics(V5B)
    failures: list[str] = []

    # ---- 1. inertness + determinism (control rows vs v5 corrected factorial) ----
    print("=== 1. Determinism vs runs/sccl_v5_fixed (v5b switches off) ===")
    for name in CONTROLS:
        a, b = v5.get(name), v5b.get(name)
        if a is None or b is None:
            failures.append(f"determinism: learner {name} missing from a run")
            continue
        diffs = []
        for k in REPORT_KEYS:
            va, vb = a["report"].get(k), b["report"].get(k)
            if va != vb:
                diffs.append(f"{k}: {va} -> {vb}")
        if a.get("frontier_score") != b.get("frontier_score"):
            diffs.append(f"frontier: {a.get('frontier_score')} -> {b.get('frontier_score')}")
        for fam in FAMILIES:
            sa, sb = fam_scores(a).get(fam), fam_scores(b).get(fam)
            if sa != sb:
                diffs.append(f"hold[{fam}]: {sa} -> {sb}")
        status = "BIT-IDENTICAL" if not diffs else "MISMATCH"
        print(f"  {name:14s} {status}" + ("" if not diffs else f"  {diffs}"))
        if diffs:
            failures.append(f"determinism: {name} differs from v5: {diffs}")

    # ---- 2. cap-probe telemetry (engagement + isolation) ----
    print("=== 2. Capability-probe engagement (artifact audit) ===")
    for name in CAP:
        L = v5b.get(name)
        if L is None:
            failures.append(f"cap telemetry: learner {name} missing")
            continue
        st = L.get("sccl", {}) or {}
        made = int(st.get("cap_probes_made", 0))
        committed = int(st.get("cap_probes_committed", 0))
        armed = int(st.get("cap_guard_armed", 0))
        gates = [((e.get("update_info") or {}).get("gate") or {})
                 for e in load_updates(V5B, name)]
        gates = [g for g in gates if g.get("method") == "sccl_rrv"]
        checked = [int(g.get("checked_cap", 0)) for g in gates]
        n_checked = sum(1 for c in checked if c > 0)
        broke = [g.get("broke_cap", []) for g in gates if g.get("broke_cap")]
        print(f"  {name:20s} probes made/committed={made}/{committed} "
              f"gates={len(gates)} with checked_cap>0: {n_checked} "
              f"cap-vetoes: {len(broke)} guard-armed updates: {armed}")
        if committed == 0:
            failures.append(f"cap telemetry: {name} committed no cap probes "
                            f"(manufacture path not engaged)")
        if gates and n_checked == 0 and committed > 0:
            failures.append(f"cap telemetry: {name} committed probes but no gate "
                            f"ever re-checked them (veto stratum not engaged)")
    for name in CONTROLS:
        if name == "frozen":
            continue
        L = v5b.get(name)
        st = (L.get("sccl", {}) or {}) if L else {}
        gates = [((e.get("update_info") or {}).get("gate") or {})
                 for e in load_updates(V5B, name)]
        leak = [g for g in gates if int(g.get("checked_cap", 0)) > 0]
        if int(st.get("cap_probes_committed", 0)) or leak:
            failures.append(f"isolation: {name} shows cap-probe activity "
                            f"(committed={st.get('cap_probes_committed')}, "
                            f"checked_cap gates={len(leak)})")
        else:
            print(f"  {name:20s} isolation OK (no cap probes, no cap stratum)")

    # ---- 3. pre-registered decision rules ----
    print("=== 3. Decision rules (pre-registered before this run) ===")
    sc = v5b["sccl"]
    cp = v5b.get("sccl_capprobe", {})
    cs = v5b.get("sccl_capprobe_strat", {})
    a_sc = fam_scores(sc)["arith"]
    a_cp = fam_scores(cp)["arith"] if cp else float("nan")
    a_cs = fam_scores(cs)["arith"] if cs else float("nan")
    f_sc = sc.get("frontier_score")
    f_cp = cp.get("frontier_score")
    f_cs = cs.get("frontier_score")
    h1 = a_cp > a_sc and f_cp >= f_sc - 0.02
    h2 = a_cs >= a_cp and f_cs >= f_sc - 0.02
    bt = a_cs >= 0.55 and f_cs >= f_sc - 0.02
    print(f"  H1  cap main   : arith {a_cp:.3f} vs sccl {a_sc:.3f}, "
          f"frontier {f_cp:+.3f} vs {f_sc:+.3f}-0.02 -> {'PASS' if h1 else 'FAIL'}")
    print(f"  H2  strat cap  : arith {a_cs:.3f} vs capprobe {a_cp:.3f}, "
          f"frontier {f_cs:+.3f} vs {f_sc:+.3f}-0.02 -> {'PASS' if h2 else 'FAIL'}")
    print(f"  BREAKTHROUGH   : arith {a_cs:.3f} >= 0.55, "
          f"frontier {f_cs:+.3f} >= {f_sc:+.3f}-0.02 -> {'PASS' if bt else 'FAIL'}")
    if bt:
        print("  -> BREAKTHROUGH rule met: multi-seed confirmation (seeds 43/44) "
              "REQUIRED before any headline claim.")

    # ---- 4. verdict table ----
    print("=== 4. Verdict table ===")
    rows = []
    header = (f"| {'learner':20s} | ACC   | BWT    | Forget | Frontier | Upd  "
              f"| arith | math  | string | drift |")
    rows.append(header)
    rows.append("|" + "-" * (len(header) - 2) + "|")
    for name in ["frozen", "sccl", "sccl_capprobe", "sccl_capprobe_strat"]:
        L = v5b.get(name)
        if L is None:
            continue
        rep, fams = L["report"], fam_scores(L)
        rows.append(
            f"| {name:20s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
            f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
            f"| {rep['updates']:>3d}  | {fams['arith']:.3f} | {fams['math_word']:.3f} "
            f"| {fams['string']:.3f}  | {fams['drift']:.3f} |")
        print(rows[-1])

    out = os.path.join(V5B, "verdict_check.md")
    with open(out, "w") as f:
        f.write("# SCCL v5b capability probes — verification & verdict\n\n")
        f.write(f"H1: {'PASS' if h1 else 'FAIL'} | H2: {'PASS' if h2 else 'FAIL'} "
                f"| BREAKTHROUGH: {'PASS' if bt else 'FAIL'}\n\n")
        f.write("\n".join(rows) + "\n\n")
        if failures:
            f.write("## FAILURES\n")
            for x in failures:
                f.write(f"- {x}\n")
        else:
            f.write("All verification checks passed.\n")
    print(f"\n[check] wrote {out}")
    if failures:
        print("[check] FAILURES:")
        for x in failures:
            print(f"  - {x}")
        sys.exit(1)
    print("[check] all checks passed")


if __name__ == "__main__":
    main()
