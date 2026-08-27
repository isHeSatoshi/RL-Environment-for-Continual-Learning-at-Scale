"""Post-run verification for SCCL v6 (Branch D, runs/sccl_v6).

Run AFTER runs/sccl_v6/metrics.json is final:
    python scripts/v6_check.py

Enforces the FAIL-CLOSED checks and PRE-REGISTERED decision rules committed in
RESEARCH_NOTES_v5.md "BRANCH D PRE-REGISTRATION" (2026-08-28, commit 34bc647),
fixed BEFORE any v6 data existed. Any fail-closed check failing => abort, do NOT
interpret the decision rules.

  C1 determinism   — the four v5b rows (frozen, sccl, sccl_capprobe,
                     sccl_capprobe_strat; seeds 42-45 unchanged) must
                     bit-reproduce runs/sccl_v5b on every computed metric.
  C2 ensemble      — the K=3 rows (sccl_capens, sccl_capens_anchor) must retain
                     >1 distinct-source-skill cap_probe in some family in the
                     vault AND log checked_cap > 1 in a gate record.
  C3 anchor        — the anchor rows (sccl_cap_anchor, sccl_capens_anchor) log
                     anchor_pen > 0 on every accepted update with lambda = 0.1;
                     the non-anchor rows log exactly 0.
  C4 isolation     — frozen/sccl commit zero cap_probes and log checked_cap = 0.
  C5 gold-free     — gcl/selfcert.py touches no gold field (AST audit), the
                     guarantee that certification (and hence every cap probe the
                     gate consumes) is spec-only.

  H1  (ensemble):     sccl_capens.arith        > sccl_capprobe_strat.arith
  H2  (anchor):       sccl_cap_anchor.arith    > sccl_capprobe_strat.arith
  H3  (composition):  sccl_capens_anchor.arith >= max(capens, cap_anchor) AND
                      best ACC among the capability cells
  BREAKTHROUGH:       sccl_capens_anchor.arith >= 0.55 AND
                      frontier >= sccl.frontier - 0.02
                      -> multi-seed (43/44) REQUIRED before any headline claim.

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
V5B = os.path.join(REPO, "runs", "sccl_v5b")
V6 = os.path.join(REPO, "runs", "sccl_v6")
DETERMINISM_ROWS = ["frozen", "sccl", "sccl_capprobe", "sccl_capprobe_strat"]
ENSEMBLE_ROWS = ["sccl_capens", "sccl_capens_anchor"]          # K=3
ANCHOR_ROWS = ["sccl_cap_anchor", "sccl_capens_anchor"]        # lambda=0.1
NONANCHOR_CAP = ["sccl_capprobe", "sccl_capprobe_strat", "sccl_capens"]
CONTROLS = ["frozen", "sccl"]
CAPABILITY_CELLS = ["sccl_capprobe", "sccl_capprobe_strat",
                    "sccl_capens", "sccl_cap_anchor", "sccl_capens_anchor"]
REPORT_KEYS = ["acc", "bwt", "fwt", "forgetting", "auc", "stability", "updates"]
FAMILIES = ["arith", "math_word", "string", "drift"]
GOLD_FIELDS = {"test_code", "reference_answer", "entry_point", "test_list", "reference"}


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


def cap_probes_in_vault(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, f"vault_{name}", "skills.json")
    if not os.path.exists(p):
        return []
    return [r for r in json.load(open(p)) if r.get("kind") == "cap_probe"]


def src(task_id: str) -> str:
    return task_id.rsplit(":c", 1)[0]


def distinct_src_per_family(caps: list) -> dict:
    d: dict = {}
    for c in caps:
        d.setdefault(c["family"], set()).add(src(c["task_id"]))
    return {f: len(s) for f, s in d.items()}


def main() -> None:
    v5b = load_metrics(V5B)
    v6 = load_metrics(V6)
    failures: list[str] = []

    # ---- C1. determinism vs runs/sccl_v5b (the four re-run rows) ----
    print("=== C1. Determinism vs runs/sccl_v5b (v5b rows re-run at same seeds) ===")
    for name in DETERMINISM_ROWS:
        a, b = v5b.get(name), v6.get(name)
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
            failures.append(f"C1 determinism: {name} differs from v5b: {diffs}")

    # ---- C2. ensemble engagement (K=3 rows) ----
    print("=== C2. Ensemble engagement (K=3 rows) ===")
    for name in ENSEMBLE_ROWS:
        L = v6.get(name)
        if L is None:
            failures.append(f"C2: learner {name} missing")
            continue
        caps = cap_probes_in_vault(V6, name)
        distinct = distinct_src_per_family(caps)
        max_distinct = max(distinct.values(), default=0)
        gates = [g for g in gates_of(load_updates(V6, name)) if g.get("method") == "sccl_rrv"]
        max_checked = max([int(g.get("checked_cap", 0)) for g in gates], default=0)
        n_multi = sum(1 for g in gates if int(g.get("checked_cap", 0)) > 1)
        print(f"  {name:22s} cap_probes={len(caps)} distinct-per-family={distinct} "
              f"max_checked_cap={max_checked} gates_with_checked_cap>1={n_multi}")
        if max_distinct < 2:
            failures.append(f"C2: {name} never held >1 distinct-source-skill cap_probe "
                            f"in a family (distinct={distinct}); ensemble not engaged")
        if max_checked < 2:
            failures.append(f"C2: {name} max checked_cap={max_checked} < 2; the veto "
                            "never re-checked more than one probe (ensemble stratum off)")

    # ---- C3. anchor engagement ----
    print("=== C3. Anchor engagement ===")
    for name in ANCHOR_ROWS:
        acc = accepted_infos(load_updates(V6, name))
        pen_pos = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
        zero_pen = len(acc) - len(pen_pos)
        print(f"  {name:22s} accepted={len(acc)} anchor_pen>0={len(pen_pos)} "
              f"lambdas={sorted(lams)}")
        if acc and len(pen_pos) != len(acc):
            failures.append(f"C3: {name} has {zero_pen}/{len(acc)} accepted updates with "
                            "anchor_pen == 0 (anchor disengaged on some updates)")
        if acc and lams != {0.1}:
            failures.append(f"C3: {name} anchor_lambda {sorted(lams)} != [0.1]")
    for name in NONANCHOR_CAP:
        acc = accepted_infos(load_updates(V6, name))
        pen_pos = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        if pen_pos:
            failures.append(f"C3: {name} (non-anchor row) logged anchor_pen>0 on "
                            f"{len(pen_pos)} updates (anchor leaked)")
        else:
            print(f"  {name:22s} anchor_pen=0 on all {len(acc)} accepted updates (correct)")

    # ---- C4. isolation (frozen/sccl) ----
    print("=== C4. Isolation (frozen/sccl) ===")
    for name in CONTROLS:
        L = v6.get(name)
        st = (L.get("sccl", {}) or {}) if L else {}
        recs = load_updates(V6, name)
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
        out = os.path.join(V6, "verdict_check.md")
        with open(out, "w") as f:
            f.write("# SCCL v6 — FAIL-CLOSED ABORT\n\n")
            for x in failures:
                f.write(f"- {x}\n")
        print(f"[check] wrote {out}")
        sys.exit(1)

    # ---- pre-registered decision rules (only if all fail-closed checks pass) ----
    print("=== Pre-registered decision rules (Branch D) ===")
    sc = v6["sccl"]
    cs = v6.get("sccl_capprobe_strat", {})
    en = v6.get("sccl_capens", {})
    an = v6.get("sccl_cap_anchor", {})
    ea = v6.get("sccl_capens_anchor", {})
    a_cs = fam_scores(cs)["arith"] if cs else float("nan")
    a_en = fam_scores(en)["arith"] if en else float("nan")
    a_an = fam_scores(an)["arith"] if an else float("nan")
    a_ea = fam_scores(ea)["arith"] if ea else float("nan")
    f_sc = sc.get("frontier_score")
    f_ea = ea.get("frontier_score")
    h1 = a_en > a_cs
    h2 = a_an > a_cs
    h3_arith = a_ea >= max(a_en, a_an)
    best_acc = max((v6.get(n, {}).get("report", {}).get("acc", -1)) for n in CAPABILITY_CELLS)
    h3_acc = ea.get("report", {}).get("acc", -1) == best_acc
    h3 = h3_arith and h3_acc
    bt = a_ea >= 0.55 and f_ea >= f_sc - 0.02
    print(f"  H1  ensemble   : arith {a_en:.3f} > capprobe_strat {a_cs:.3f} "
          f"-> {'PASS' if h1 else 'FAIL'}")
    print(f"  H2  anchor     : arith {a_an:.3f} > capprobe_strat {a_cs:.3f} "
          f"-> {'PASS' if h2 else 'FAIL'}")
    print(f"  H3  composition: arith {a_ea:.3f} >= max({a_en:.3f},{a_an:.3f}) "
          f"AND acc==best({best_acc:.3f}) -> {'PASS' if h3 else 'FAIL'}")
    print(f"  BREAKTHROUGH   : arith {a_ea:.3f} >= 0.55 AND frontier {f_ea:+.3f} "
          f">= {f_sc:+.3f}-0.02 -> {'PASS' if bt else 'FAIL'}")
    if bt:
        print("  -> BREAKTHROUGH rule met: multi-seed confirmation (seeds 43/44) "
              "REQUIRED before any headline claim.")

    # ---- verdict table ----
    print("=== Verdict table ===")
    rows = []
    header = (f"| {'learner':22s} | ACC   | BWT    | Forget | Frontier | Upd  "
              f"| arith | math  | string | drift |")
    rows.append(header)
    rows.append("|" + "-" * (len(header) - 2) + "|")
    for name in ["frozen", "sccl"] + CAPABILITY_CELLS:
        L = v6.get(name)
        if L is None:
            continue
        rep, fams = L["report"], fam_scores(L)
        rows.append(
            f"| {name:22s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
            f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
            f"| {rep['updates']:>3d}  | {fams['arith']:.3f} | {fams['math_word']:.3f} "
            f"| {fams['string']:.3f}  | {fams['drift']:.3f} |")
        print(rows[-1])

    out = os.path.join(V6, "verdict_check.md")
    with open(out, "w") as f:
        f.write("# SCCL v6 (Branch D) capability ensembles + anchor — verdict\n\n")
        f.write(f"H1: {'PASS' if h1 else 'FAIL'} | H2: {'PASS' if h2 else 'FAIL'} | "
                f"H3: {'PASS' if h3 else 'FAIL'} | BREAKTHROUGH: "
                f"{'PASS' if bt else 'FAIL'}\n\n")
        f.write("All fail-closed checks (C1-C5) passed.\n\n")
        f.write("\n".join(rows) + "\n")
    print(f"\n[check] wrote {out}")
    print("[check] all fail-closed checks passed")


if __name__ == "__main__":
    main()
