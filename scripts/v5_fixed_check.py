"""Post-run verification for the CORRECTED v5 factorial (runs/sccl_v5_fixed).

Run AFTER runs/sccl_v5_fixed/metrics.json is final:
    python scripts/v5_fixed_check.py

Checks, in order:
  1. DETERMINISM ACROSS THE ENGINE FIX — the non-anchor rows (frozen, sccl,
     sccl_strat, vsr_nogold) must bit-reproduce the first flight's
     (runs/sccl_v5) values on every headline metric and per-family holdout.
     The anchor fix only touches the anchor_lambda > 0 path; any mismatch
     here means the fix leaked into the shared code path.
  2. ANCHOR ENGAGEMENT — every accepted update of the two anchor learners
     must log anchor_pen > 0 in the trajectory (anchor_pen == 0.0 everywhere
     is the exact signature of the v4 no-op bug); anchor_lambda must be 0.1.
     Conversely the four non-anchor learners must log anchor_pen == 0.0
     (isolation).
  3. PRE-REGISTERED DECISION RULES (RESEARCH_NOTES_v5.md), now measured with
     an engaged anchor:
       H1  (coverage):     sccl_strat.arith > sccl.arith AND
                           sccl_strat.frontier >= sccl.frontier - 0.02
       H2  (composition):  sccl_anchor_strat.arith > sccl.arith AND high ACC
       BREAKTHROUGH:       sccl_anchor_strat.arith >= 0.55 AND
                           sccl_anchor_strat.frontier >= sccl.frontier - 0.02
  4. VERDICT TABLE + arith-holdout per family (mean score, 0.2 partial credit)
     written to runs/sccl_v5_fixed/verdict_check.md.

Gold labels never enter any check here except as the evaluation metric
(holdout scores already recorded by the runner at measurement time).
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST = os.path.join(REPO, "runs", "sccl_v5")
FIXED = os.path.join(REPO, "runs", "sccl_v5_fixed")
NON_ANCHOR = ["frozen", "sccl", "sccl_strat", "vsr_nogold"]
ANCHOR = ["sccl_anchor_lo", "sccl_anchor_strat"]
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
    first = load_metrics(FIRST)
    fixed = load_metrics(FIXED)
    failures: list[str] = []

    # ---- 1. determinism across the engine fix (non-anchor rows) ----
    print("=== 1. Determinism across the anchor fix (non-anchor rows) ===")
    for name in NON_ANCHOR:
        a, b = first.get(name), fixed.get(name)
        if a is None or b is None:
            failures.append(f"determinism: learner {name} missing from a run")
            continue
        diffs = []
        for k in REPORT_KEYS:
            va, vb = a["report"].get(k), b["report"].get(k)
            if va != vb:
                diffs.append(f"{k}: {va} -> {vb}")
        fa = a.get("frontier_score")
        fb = b.get("frontier_score")
        if fa != fb:
            diffs.append(f"frontier: {fa} -> {fb}")
        for fam in FAMILIES:
            sa, sb = fam_scores(a).get(fam), fam_scores(b).get(fam)
            if sa != sb:
                diffs.append(f"hold[{fam}]: {sa} -> {sb}")
        status = "BIT-IDENTICAL" if not diffs else "MISMATCH"
        print(f"  {name:14s} {status}" + ("" if not diffs else f"  {diffs}"))
        if diffs:
            failures.append(f"determinism: {name} differs across the fix: {diffs}")

    # ---- 2. anchor engagement from run artifacts ----
    print("=== 2. Anchor engagement (anchor_pen audit trail) ===")
    for name in ANCHOR:
        ups = [e.get("update_info", {}) for e in load_updates(FIXED, name)]
        ups = [u for u in ups if u.get("executed")]
        pens = [u.get("anchor_pen", 0.0) for u in ups if u.get("accepted")]
        # anchor_lambda is surfaced only on ACCEPTED updates (rejected ones
        # never carry it), so restrict the lambda audit to the same set.
        lams = {u.get("anchor_lambda", 0.0) for u in ups if u.get("accepted")}
        if not pens:
            failures.append(f"engagement: {name} has no accepted updates")
            continue
        n_zero = sum(1 for p in pens if p == 0.0)
        print(f"  {name:18s} accepted={len(pens)} anchor_pen>0: {len(pens) - n_zero} "
              f"lambdas={sorted(lams)} pen[min/med/max]="
              f"{min(pens):.3g}/{sorted(pens)[len(pens)//2]:.3g}/{max(pens):.3g}")
        if n_zero:
            failures.append(f"engagement: {name} logged anchor_pen==0 on "
                            f"{n_zero}/{len(pens)} accepted updates (v4 no-op signature)")
        if lams != {0.1}:
            failures.append(f"engagement: {name} anchor_lambda {lams} != {{0.1}}")
    for name in NON_ANCHOR:
        if name == "frozen":
            continue
        ups = [e.get("update_info", {}) for e in load_updates(FIXED, name)]
        bad = [u for u in ups if u.get("executed") and u.get("anchor_pen", 0.0) != 0.0]
        if bad:
            failures.append(f"isolation: {name} has {len(bad)} updates with anchor_pen != 0")
        else:
            print(f"  {name:18s} isolation OK (anchor_pen == 0 on all updates)")

    # ---- 3. pre-registered decision rules ----
    print("=== 3. Decision rules (engaged anchor) ===")
    sc = fixed["sccl"]["report"]
    st = fixed["sccl_strat"]
    bo = fixed["sccl_anchor_strat"]
    a_sc = fam_scores(fixed["sccl"])["arith"]
    a_st = fam_scores(st)["arith"]
    a_bo = fam_scores(bo)["arith"]
    f_sc, f_st, f_bo = (fixed["sccl"].get("frontier_score"),
                        st.get("frontier_score"), bo.get("frontier_score"))
    h1 = a_st > a_sc and f_st >= f_sc - 0.02
    h2 = a_bo > a_sc and bo["report"]["acc"] >= sc["acc"] - 0.02
    bt = a_bo >= 0.55 and f_bo >= f_sc - 0.02
    print(f"  H1  coverage    : arith {a_st:.3f} vs sccl {a_sc:.3f}, "
          f"frontier {f_st:+.3f} vs {f_sc:+.3f}-0.02 -> {'PASS' if h1 else 'FAIL'}")
    print(f"  H2  composition : arith {a_bo:.3f} vs sccl {a_sc:.3f}, "
          f"ACC {bo['report']['acc']:.3f} vs {sc['acc']:.3f} -> {'PASS' if h2 else 'FAIL'}")
    print(f"  BREAKTHROUGH    : arith {a_bo:.3f} >= 0.55, "
          f"frontier {f_bo:+.3f} >= {f_sc:+.3f}-0.02 -> {'PASS' if bt else 'FAIL'}")
    if bt:
        print("  -> BREAKTHROUGH rule met: run scripts/v5_multiseed.py (seeds 43/44) "
              "BEFORE any headline claim.")

    # ---- 4. verdict table ----
    print("=== 4. Verdict table ===")
    rows = []
    header = (f"| {'learner':18s} | ACC   | BWT    | Forget | Frontier | Upd  "
              f"| arith | math  | string | drift |")
    rows.append(header)
    rows.append("|" + "-" * (len(header) - 2) + "|")
    for name in ["frozen", "sccl", "sccl_strat", "sccl_anchor_lo",
                 "sccl_anchor_strat", "vsr_nogold"]:
        L = fixed.get(name)
        if L is None:
            continue
        rep, fams = L["report"], fam_scores(L)
        rows.append(
            f"| {name:18s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
            f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
            f"| {rep['updates']:>3d}  | {fams['arith']:.3f} | {fams['math_word']:.3f} "
            f"| {fams['string']:.3f}  | {fams['drift']:.3f} |")
        print(rows[-1])

    out = os.path.join(FIXED, "verdict_check.md")
    with open(out, "w") as f:
        f.write("# Corrected v5 factorial — verification & verdict\n\n")
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
