"""Post-run verification for SCCL v7 (Branch E, E1 — runs/sccl_v7).

Run AFTER runs/sccl_v7/metrics.json is final:
    python scripts/v7_check.py

Enforces the FAIL-CLOSED checks and PRE-REGISTERED decision rules committed in
RESEARCH_NOTES_v5.md "BRANCH E PRE-REGISTRATION" (commit c53c3aa, corrected
99bfaf7 BEFORE any v7 run). Any fail-closed check failing => abort, do NOT
interpret the decision rules.

  C1 determinism   — the four v6-prefix rows (frozen, sccl, sccl_capprobe,
                     sccl_capprobe_strat; seeds 42-45 unchanged) must
                     bit-reproduce runs/sccl_v6 on every computed metric.
  C2 engagement    — every theta>0 row logs cap_rates on EVERY gate with
                     checked_cap>0 (a checked gate with empty cap_rates is a
                     telemetry hole => abort).
  C3 dose sanity   — on theta>0 rows the gate records carry the configured
                     theta (cap_retain_min) and cap_n=3, and every cap_rates
                     denominator is 3 (or 3+cap_margin=5 when the bounded-
                     damage guard armed).
  C4 isolation     — frozen/sccl commit zero cap_probes and log checked_cap=0;
                     the theta=0 cap row (sccl_capprobe_strat) logs NO E1
                     fields (cap_rates/cap_retain_min/cap_n) on any gate.
  C5 gold-free     — gcl/selfcert.py touches no gold field (AST audit), the
                     guarantee that certification (and hence every cap probe
                     the gate consumes) is spec-only.

  H1  (sensitivity): sccl_strict.arith > strat.arith OR
                     sccl_majority.arith > strat.arith (strat = in-ladder
                     sccl_capprobe_strat baseline, 0.375 in v6).
                     VOID if H1b shows strict never vetoed more than baseline.
  H1b (dose order) : cap-veto counts strict >= majority >= baseline.
  H2  (width+sens) : sccl_ens_strict.arith >= sccl_strict.arith.
  H3  (composition): sccl_ens_strict_anchor.arith >= max(rows 4..6) AND
                     its ACC >= max ACC of rows 4..6 (degenerate rows
                     excluded from the maxima).
  BREAKTHROUGH     : sccl_ens_strict_anchor.arith >= 0.55 AND
                     frontier >= sccl.frontier - 0.02
                     -> multi-seed REQUIRED before any headline claim.

  DEGENERACY GUARD : a theta>0 row with < 3 accepted updates is DEGENERATE
                     (veto stall, not mechanism success) and is excluded from
                     the H2/H3 maxima; a degenerate breakthrough cell forces
                     BREAKTHROUGH = FAIL.

Gold labels never enter any check except as the evaluation metric (holdout
scores already recorded by the runner at measurement time) and the post-hoc
telemetry it writes.
"""
from __future__ import annotations

import ast
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.join(REPO, "runs", "sccl_v6")
V7 = os.path.join(REPO, "runs", "sccl_v7")
DETERMINISM_ROWS = ["frozen", "sccl", "sccl_capprobe", "sccl_capprobe_strat"]
THETA_ROWS = {  # name -> configured theta (must equal gate cap_retain_min)
    "sccl_strict": 1.0,
    "sccl_majority": 2 / 3,
    "sccl_ens_strict": 1.0,
    "sccl_ens_strict_anchor": 1.0,
}
THETA0_CAP_ROW = "sccl_capprobe_strat"
ANCHOR_ROW = "sccl_ens_strict_anchor"
NONANCHOR_CAP = ["sccl_capprobe", "sccl_capprobe_strat",
                 "sccl_strict", "sccl_majority", "sccl_ens_strict"]
CONTROLS = ["frozen", "sccl"]
E1_CELLS = ["sccl_strict", "sccl_majority", "sccl_ens_strict", "sccl_ens_strict_anchor"]
REPORT_KEYS = ["acc", "bwt", "fwt", "forgetting", "auc", "stability", "updates"]
FAMILIES = ["arith", "math_word", "string", "drift"]
GOLD_FIELDS = {"test_code", "reference_answer", "entry_point", "test_list", "reference"}
CAP_N = 3
CAP_MARGIN = 2
DEGENERATE_MIN_UPDATES = 3


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


def update_infos(recs: list) -> list:
    return [e["update_info"] for e in recs if e.get("update_info")]


def gates_of(recs: list) -> list:
    return [g for g in (ui.get("gate") for ui in update_infos(recs)) if g]


def accepted_infos(recs: list) -> list:
    return [ui for ui in update_infos(recs) if ui.get("accepted")]


def cap_gates(recs: list) -> list:
    return [g for g in gates_of(recs)
            if g.get("method") == "sccl_rrv" and int(g.get("checked_cap", 0)) > 0]


def cap_vetoes(recs: list) -> int:
    return sum(1 for g in gates_of(recs)
               if g.get("veto") and g.get("broke_cap"))


def main() -> None:
    v6 = load_metrics(V6)
    v7 = load_metrics(V7)
    failures: list[str] = []

    # ---- C1. determinism vs runs/sccl_v6 (the four prefix rows) ----
    print("=== C1. Determinism vs runs/sccl_v6 (v6 prefix re-run at same seeds) ===")
    for name in DETERMINISM_ROWS:
        a, b = v6.get(name), v7.get(name)
        if a is None or b is None:
            failures.append(f"C1 determinism: learner {name} missing from a run")
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
        print(f"  {name:22s} {status}" + ("" if not diffs else f"  {diffs}"))
        if diffs:
            failures.append(f"C1 determinism: {name} differs from v6: {diffs}")

    # ---- C2. E1 engagement: cap_rates on every checked gate of theta rows ----
    print("=== C2. E1 engagement (cap_rates on every checked gate) ===")
    for name in THETA_ROWS:
        cg = cap_gates(load_updates(V7, name))
        holes = [g for g in cg if not g.get("cap_rates")]
        with_rates = len(cg) - len(holes)
        pct = 100.0 * with_rates / len(cg) if cg else 0.0
        print(f"  {name:22s} checked-cap gates={len(cg)} with cap_rates={with_rates} "
              f"({pct:.0f}%)")
        if not cg:
            failures.append(f"C2: {name} has no gate with checked_cap>0 "
                            "(cap stratum never engaged)")
        if holes:
            failures.append(f"C2: {name} has {len(holes)} checked-cap gates with "
                            "EMPTY cap_rates (telemetry hole)")

    # ---- C3. dose sanity: logged theta == configured, cap_n == 3, rates n in {3,5} ----
    print("=== C3. Dose sanity (theta/cap_n logged, rate denominators) ===")
    for name, theta in THETA_ROWS.items():
        cg = cap_gates(load_updates(V7, name))
        bad_theta = [g for g in cg
                     if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
        bad_n = [g for g in cg if int(g.get("cap_n", -1)) != CAP_N]
        denoms = sorted({int(r["n"]) for g in cg
                         for r in (g.get("cap_rates") or {}).values()})
        legal = {CAP_N, CAP_N + CAP_MARGIN}
        bad_denoms = [d for d in denoms if d not in legal]
        print(f"  {name:22s} theta_gates_mismatched={len(bad_theta)} "
              f"cap_n_mismatched={len(bad_n)} rate_denominators={denoms}")
        if bad_theta:
            failures.append(f"C3: {name} logged cap_retain_min != {theta} on "
                            f"{len(bad_theta)} gates")
        if bad_n:
            failures.append(f"C3: {name} logged cap_n != {CAP_N} on {len(bad_n)} gates")
        if bad_denoms:
            failures.append(f"C3: {name} cap_rates denominators {bad_denoms} outside "
                            f"legal set {sorted(legal)}")

    # ---- C4. isolation (frozen/sccl; legacy gate shape on theta=0 cap row) ----
    print("=== C4. Isolation (frozen/sccl; theta=0 row keeps legacy gate shape) ===")
    for name in CONTROLS:
        L = v7.get(name)
        st = (L.get("sccl", {}) or {}) if L else {}
        recs = load_updates(V7, name)
        committed = int(st.get("cap_probes_committed", 0))
        checked = [g for g in gates_of(recs) if int(g.get("checked_cap", 0)) > 0]
        anchor = [ui for ui in accepted_infos(recs)
                  if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        if committed or checked or anchor:
            failures.append(f"C4 isolation: {name} shows cap/anchor activity "
                            f"(committed={committed}, checked_cap_gates={len(checked)}, "
                            f"anchor_updates={len(anchor)})")
        else:
            print(f"  {name:22s} isolation OK (no cap probes, no cap stratum, no anchor)")
    recs = load_updates(V7, THETA0_CAP_ROW)
    cg = cap_gates(recs)
    e1_leak = [g for g in gates_of(recs)
               if ("cap_rates" in g) or ("cap_retain_min" in g) or ("cap_n" in g)]
    if not cg:
        failures.append(f"C4: {THETA0_CAP_ROW} never checked cap probes "
                        "(cap stratum disengaged on the baseline row)")
    if e1_leak:
        failures.append(f"C4: {THETA0_CAP_ROW} (theta=0) logged E1 fields on "
                        f"{len(e1_leak)} gates (legacy gate shape violated)")
    else:
        print(f"  {THETA0_CAP_ROW:22s} theta=0 legacy shape OK "
              f"(checked-cap gates={len(cg)}, no E1 fields)")

    # ---- C4b. anchor engagement on the composition row; no leak elsewhere ----
    print("=== C4b. Anchor engagement ===")
    acc = accepted_infos(load_updates(V7, ANCHOR_ROW))
    pen_pos = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
    lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
    print(f"  {ANCHOR_ROW:22s} accepted={len(acc)} anchor_pen>0={len(pen_pos)} "
          f"lambdas={sorted(lams)}")
    if acc and len(pen_pos) != len(acc):
        failures.append(f"C4b: {ANCHOR_ROW} has {len(acc) - len(pen_pos)}/{len(acc)} "
                        "accepted updates with anchor_pen == 0 (anchor disengaged)")
    if acc and lams != {0.1}:
        failures.append(f"C4b: {ANCHOR_ROW} anchor_lambda {sorted(lams)} != [0.1]")
    for name in NONANCHOR_CAP:
        n_leak = sum(1 for ui in accepted_infos(load_updates(V7, name))
                     if float(ui.get("anchor_pen", 0.0) or 0.0) > 0)
        if n_leak:
            failures.append(f"C4b: {name} (non-anchor row) logged anchor_pen>0 on "
                            f"{n_leak} updates (anchor leaked)")

    # ---- C5. gold-free (certifier never reads gold fields) ----
    print("=== C5. Gold-free (AST audit of gcl/selfcert.py) ===")
    path = os.path.join(REPO, "gcl", "selfcert.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    violations = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and n.attr in GOLD_FIELDS]
    if violations:
        failures.append(f"C5 gold-free: selfcert.py touches gold fields: {violations}")
        print(f"  VIOLATIONS: {violations}")
    else:
        print("  gcl/selfcert.py touches no gold field (certification is spec-only)")

    if failures:
        print("\n[check] FAIL-CLOSED check(s) FAILED — aborting before decision rules:")
        for f in failures:
            print(f"  - {f}")
        out = os.path.join(V7, "verdict_check.md")
        with open(out, "w") as f:
            f.write("# SCCL v7 — FAIL-CLOSED ABORT\n\n")
            for x in failures:
                f.write(f"- {x}\n")
        print(f"[check] wrote {out}")
        sys.exit(1)

    # ---- degeneracy guard (pre-registered): <3 accepted updates = veto stall ----
    degenerate = set()
    for name in E1_CELLS:
        upd = v7.get(name, {}).get("report", {}).get("updates", 0)
        if int(upd) < DEGENERATE_MIN_UPDATES:
            degenerate.add(name)
            print(f"  [degeneracy] {name}: {upd} accepted updates < "
                  f"{DEGENERATE_MIN_UPDATES} -> DEGENERATE (veto stall), excluded "
                  "from H2/H3 maxima")

    # ---- pre-registered decision rules (only if all fail-closed checks pass) ----
    print("=== Pre-registered decision rules (Branch E) ===")
    sc = v7["sccl"]
    strat = v7.get(THETA0_CAP_ROW, {})
    strict = v7.get("sccl_strict", {})
    majority = v7.get("sccl_majority", {})
    ens = v7.get("sccl_ens_strict", {})
    comp = v7.get(ANCHOR_ROW, {})
    a_strat = fam_scores(strat)["arith"] if strat else float("nan")
    a_strict = fam_scores(strict)["arith"] if strict else float("nan")
    a_majority = fam_scores(majority)["arith"] if majority else float("nan")
    a_ens = fam_scores(ens)["arith"] if ens else float("nan")
    a_comp = fam_scores(comp)["arith"] if comp else float("nan")
    f_sc = sc.get("frontier_score")
    f_comp = comp.get("frontier_score")

    # H1b first: dose ordering of cap-veto counts (strict >= majority >= baseline)
    v_base = cap_vetoes(load_updates(V7, THETA0_CAP_ROW))
    v_strict = cap_vetoes(load_updates(V7, "sccl_strict"))
    v_majority = cap_vetoes(load_updates(V7, "sccl_majority"))
    h1b = v_strict >= v_majority >= v_base
    print(f"  H1b dose order : cap-vetoes strict={v_strict} >= majority={v_majority} "
          f">= baseline={v_base} -> {'PASS' if h1b else 'FAIL'}")
    h1_void = v_strict < v_base
    if h1_void:
        print("  [H1b] strict vetoed FEWER than baseline: theta never engaged its "
              "extra sensitivity -> H1 verdict VOID, re-audit via cap_rates")

    h1 = (a_strict > a_strat or a_majority > a_strat) and not h1_void
    cells = [n for n in ["sccl_strict", "sccl_majority", "sccl_ens_strict"]
             if n not in degenerate]
    cell_arith = {n: fam_scores(v7[n])["arith"] for n in cells}
    cell_acc = {n: v7[n]["report"]["acc"] for n in cells}
    h2 = a_ens >= a_strict
    h3_arith = (comp is not None and cells and
                a_comp >= max(cell_arith.values()))
    h3_acc = (comp is not None and cells and
              comp.get("report", {}).get("acc", -1) >= max(cell_acc.values()))
    h3 = h3_arith and h3_acc
    bt = (ANCHOR_ROW not in degenerate and a_comp >= 0.55
          and f_comp is not None and f_sc is not None and f_comp >= f_sc - 0.02)
    print(f"  H1  sensitivity: arith strict={a_strict:.3f} / majority={a_majority:.3f} "
          f"> baseline={a_strat:.3f} -> "
          f"{'VOID' if h1_void else ('PASS' if h1 else 'FAIL')}")
    print(f"  H2  width+sens  : arith ens_strict={a_ens:.3f} >= strict={a_strict:.3f} "
          f"-> {'PASS' if h2 else 'FAIL'}")
    if cells:
        print(f"  H3  composition : arith {a_comp:.3f} >= max({cell_arith}) AND "
              f"acc {comp.get('report', {}).get('acc', -1):.3f} >= max({cell_acc}) -> "
              f"{'PASS' if h3 else 'FAIL'}"
              + (" [degenerate rows excluded]" if degenerate else ""))
    else:
        print("  H3  composition : no non-degenerate cells -> FAIL")
    print(f"  BREAKTHROUGH    : arith {a_comp:.3f} >= 0.55 AND frontier {f_comp:+.3f} "
          f">= {f_sc:+.3f}-0.02 -> {'PASS' if bt else 'FAIL'}"
          + (f" [{ANCHOR_ROW} DEGENERATE]" if ANCHOR_ROW in degenerate else ""))
    if bt:
        print("  -> BREAKTHROUGH rule met: multi-seed confirmation REQUIRED "
              "before any headline claim.")

    # ---- verdict table ----
    print("=== Verdict table ===")
    rows = []
    header = (f"| {'learner':22s} | ACC   | BWT    | Forget | Frontier | Upd  "
              f"| arith | math  | string | drift |")
    rows.append(header)
    rows.append("|" + "-" * (len(header) - 2) + "|")
    for name in DETERMINISM_ROWS + E1_CELLS:
        L = v7.get(name)
        if L is None:
            continue
        rep, fams = L["report"], fam_scores(L)
        flag = " *" if name in degenerate else ""
        rows.append(
            f"| {name + flag:22s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
            f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
            f"| {rep['updates']:>3d}  | {fams['arith']:.3f} | {fams['math_word']:.3f} "
            f"| {fams['string']:.3f}  | {fams['drift']:.3f} |")
        print(rows[-1])

    out = os.path.join(V7, "verdict_check.md")
    with open(out, "w") as f:
        f.write("# SCCL v7 (Branch E, E1 pass-rate margin veto) — verdict\n\n")
        f.write(f"H1: {'VOID' if h1_void else ('PASS' if h1 else 'FAIL')} | "
                f"H1b: {'PASS' if h1b else 'FAIL'} | "
                f"H2: {'PASS' if h2 else 'FAIL'} | "
                f"H3: {'PASS' if h3 else 'FAIL'} | BREAKTHROUGH: "
                f"{'PASS' if bt else 'FAIL'}\n\n")
        f.write("All fail-closed checks (C1-C5) passed.\n")
        if degenerate:
            f.write(f"Degenerate rows (veto stall, <{DEGENERATE_MIN_UPDATES} "
                    f"updates): {sorted(degenerate)}\n")
        f.write("\n" + "\n".join(rows) + "\n")
    print(f"\n[check] wrote {out}")
    print("[check] all fail-closed checks passed")


if __name__ == "__main__":
    main()
