"""Post-run verification for SCCL v8 (Branch F, G1 — runs/sccl_v8).

Run AFTER runs/sccl_v8/metrics.json is final:
    python scripts/v8_check.py

Enforces the FAIL-CLOSED checks and PRE-REGISTERED decision rules committed in
RESEARCH_NOTES_v5.md "BRANCH F PRE-REGISTRATION — G1 GENERALIZATION WITNESSES
(v8 ladder)" (BEFORE any v8 implementation). Any fail-closed check failing =>
abort, do NOT interpret the decision rules.

  C1 determinism   — the four prefix rows (frozen, sccl, sccl_capprobe,
                     sccl_capprobe_strat; seeds 42-45 unchanged) must
                     bit-reproduce runs/sccl_v7 idx0-3 on every computed
                     metric, frontier, and family holdout score.
  C2 engagement    — each gen row manufactures >= 1 gen probe (sccl_gen_probe
                     first-contact markers), checked gates show ':g' ids in
                     checked_cap_ids, theta>0 rows log cap_rates covering at
                     least one ':g' id, and logged cap_retain_min/cap_n ==
                     config on every checked gate.
  C3 retirement    — no ':g' probe whose source task T is checked at a gate
                     during or after T's own training episode (trajectory
                     walk), and every ':g' source is a stream TRAINING task —
                     disjoint from the gold holdout task ids in
                     eval_detail.final_heldout (a holdout-sourced probe is
                     gold leakage and fails the run).
  C4 isolation     — prefix rows show ZERO gen activity; the theta=0 gen row
                     (sccl_genprobe) logs no E1 fields; the anchor row logs
                     anchor_pen > 0 with lambda = 0.1 on every accepted
                     update, and no other row logs anchor activity.
  C5 gold-free     — AST audit: gcl/selfcert.py touches no gold field; the
                     _gen_make manufacture path in gcl/experiment.py consumes
                     no gold label field (entry_point counted as gold when
                     sccl_derive_entry is on, i.e. it is never read);
                     gcl/vault.py touches no gold field.

  H1  (coverage main effect): sccl_genprobe.arith > 0.375 (the reproducible
                     v5b/v6/v7 capability baseline under the SAME any-pass
                     rule; the only change is witness coverage).
  H2  (coverage+sensitivity): sccl_genprobe_strict.arith >= max(0.375,
                     genprobe.arith) AND accepted updates >= 5 (anti-stall).
  H3  (composition): sccl_genprobe_strict_anchor arith AND ACC >= max of the
                     gen rows (degenerate rows excluded from the maxima).
  M1  (mechanism, recorded, not gated): gen-witness sensitivity — fraction of
                     probe-checked gold-erosion accepted updates whose gate
                     shows a ':g' break or a ':g' pooled rate < theta.
                     v7 baseline 0%; target > 50%.
  BREAKTHROUGH     : sccl_genprobe_strict_anchor arith >= 0.55 AND frontier
                     >= sccl.frontier - 0.02 AND accepted updates >= 5
                     -> multi-seed REQUIRED before any headline claim.

  DEGENERACY GUARD : a theta>0 gen row with < 5 accepted updates is
                     DEGENERATE (veto stall / witness too harsh — a dose
                     finding) and is excluded from the H2/H3 maxima; a
                     degenerate breakthrough cell forces BREAKTHROUGH = FAIL.

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
V7 = os.path.join(REPO, "runs", "sccl_v7")
V8 = os.path.join(REPO, "runs", "sccl_v8")
CONFIG = os.path.join(REPO, "configs", "sccl_v8.json")
DETERMINISM_ROWS = ["frozen", "sccl", "sccl_capprobe", "sccl_capprobe_strat"]
GEN_ROWS = ["sccl_genprobe", "sccl_genprobe_strict", "sccl_genprobe_strict_anchor"]
THETA_ROWS = {  # name -> configured theta (must equal gate cap_retain_min)
    "sccl_genprobe_strict": 1.0,
    "sccl_genprobe_strict_anchor": 1.0,
}
THETA0_GEN_ROW = "sccl_genprobe"
ANCHOR_ROW = "sccl_genprobe_strict_anchor"
NONANCHOR_ROWS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat",
                  "sccl_genprobe", "sccl_genprobe_strict"]
REPORT_KEYS = ["acc", "bwt", "fwt", "forgetting", "auc", "stability", "updates"]
FAMILIES = ["arith", "math_word", "string", "drift"]
LABEL_FIELDS = {"test_code", "reference_answer", "test_list", "reference"}
GOLD_FIELDS = LABEL_FIELDS | {"entry_point"}
CAP_N = 3
CAP_MARGIN = 2
BASELINE_ARITH = 0.375
DEGENERATE_MIN_UPDATES = 5
M1_TARGET = 0.50
EPS = 0.05


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


def holdout_ids(learners: dict) -> set:
    ids = set()
    for L in learners.values():
        for fam, recs in (L.get("eval_detail", {})
                          .get("final_heldout", {}) or {}).items():
            for t in recs or []:
                if t.get("task_id"):
                    ids.add(t["task_id"])
    return ids


def load_records(run_dir: str, name: str) -> list:
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


def gen_probe_rate(g: dict) -> float | None:
    """Pooled pass rate over the ':g' entries of one gate's cap_rates."""
    rates = g.get("cap_rates") or {}
    gen = {k: v for k, v in rates.items() if k.endswith(":g")}
    if not gen:
        return None
    passes = sum(int(v.get("passes", 0)) for v in gen.values())
    n = sum(int(v.get("n", 0)) for v in gen.values())
    return passes / n if n > 0 else None


def main() -> None:
    v7 = load_metrics(V7)
    v8 = load_metrics(V8)
    failures: list[str] = []
    HOLD = holdout_ids(v8)

    # ---- C1. determinism vs runs/sccl_v7 (the four prefix rows) ----
    print("=== C1. Determinism vs runs/sccl_v7 (prefix re-run at same seeds) ===")
    for name in DETERMINISM_ROWS:
        a, b = v7.get(name), v8.get(name)
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
            failures.append(f"C1 determinism: {name} differs from v7: {diffs}")

    # ---- C2. G1 engagement (manufacture + ':g' in gates + dose) ----
    print("=== C2. G1 engagement (manufacture at first contact, ':g' in gates) ===")
    for name in GEN_ROWS:
        recs = load_records(V8, name)
        markers = [r for r in recs if r.get("sccl_gen_probe")]
        cg = cap_gates(recs)
        gen_gates = [g for g in cg
                     if any(i.endswith(":g") for i in g.get("checked_cap_ids", []))]
        st = (v8.get(name, {}).get("sccl", {}) or {})
        made = int(st.get("gen_probes_committed", 0))
        retired = int(st.get("gen_probes_retired", 0))
        print(f"  {name:22s} first-contact markers={len(markers)} "
              f"gen probes committed={made} retired={retired} "
              f"checked-cap gates={len(cg)} with-':g'={len(gen_gates)}")
        if not markers or made < 1:
            failures.append(f"C2: {name} manufactured no gen probe "
                            f"(markers={len(markers)}, committed={made})")
        if not gen_gates:
            failures.append(f"C2: {name} never checked a ':g' probe at any gate")
        if name in THETA_ROWS:
            theta = THETA_ROWS[name]
            bad_theta = [g for g in cg
                         if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
            bad_n = [g for g in cg if int(g.get("cap_n", -1)) != CAP_N]
            holes = [g for g in cg if not g.get("cap_rates")]
            denoms = sorted({int(r["n"]) for g in cg
                             for r in (g.get("cap_rates") or {}).values()})
            legal = {CAP_N, CAP_N + CAP_MARGIN}
            bad_denoms = [d for d in denoms if d not in legal]
            gen_rate_ids = [tid for g in cg for tid in (g.get("cap_rates") or {})
                            if tid.endswith(":g")]
            print(f"  {name:22s} theta_mismatched={len(bad_theta)} "
                  f"cap_n_mismatched={len(bad_n)} rate_holes={len(holes)} "
                  f"denominators={denoms} cap_rates ':g' ids={len(gen_rate_ids)}")
            if bad_theta:
                failures.append(f"C2: {name} logged cap_retain_min != {theta} on "
                                f"{len(bad_theta)} gates")
            if bad_n:
                failures.append(f"C2: {name} logged cap_n != {CAP_N} on "
                                f"{len(bad_n)} gates")
            if holes:
                failures.append(f"C2: {name} has {len(holes)} checked-cap gates "
                                "with EMPTY cap_rates (telemetry hole)")
            if bad_denoms:
                failures.append(f"C2: {name} cap_rates denominators {bad_denoms} "
                                f"outside legal set {sorted(legal)}")
            if gen_gates and not gen_rate_ids:
                failures.append(f"C2: {name} checked ':g' probes but cap_rates "
                                "never covered a ':g' id")

    # ---- C3. retirement + anti-contamination (trajectory walk) ----
    print("=== C3. Retirement/anti-contamination (no ':g' checked during/after "
          "source training; no holdout sources) ===")
    for name in GEN_ROWS:
        recs = load_records(V8, name)
        trained, bad_retire, bad_hold, checked_srcs = set(), [], [], set()
        for r in recs:
            # count the record's own task as in-training BEFORE inspecting its
            # gate: a ':g' probe sourced from the currently-training task is a
            # retirement violation ("during or after T's own episode").
            if r.get("task_id"):
                trained.add(r["task_id"])
            g = (r.get("update_info") or {}).get("gate") or {}
            for i in g.get("checked_cap_ids", []):
                if not i.endswith(":g"):
                    continue
                src = i[:-2]
                checked_srcs.add(src)
                if src in trained:
                    bad_retire.append((r.get("task_id"), i))
                if src in HOLD:
                    bad_hold.append(i)
            msrc = (r.get("sccl_gen_probe") or {}).get("src")
            if msrc:
                if msrc in HOLD:
                    bad_hold.append(msrc)
                if msrc in trained:
                    bad_retire.append((r.get("task_id"), msrc + ":g (manufactured)"))
        print(f"  {name:22s} distinct ':g' sources checked={len(checked_srcs)} "
              f"retirement violations={len(bad_retire)} "
              f"holdout-sourced={len(set(bad_hold))}")
        if bad_retire:
            failures.append(f"C3 retirement: {name} checked a gen probe whose "
                            f"source was already trained: {bad_retire[:3]}")
        if bad_hold:
            failures.append(f"C3 contamination: {name} gen probes sourced from "
                            f"GOLD HOLDOUT tasks: {sorted(set(bad_hold))}")

    # ---- C4. isolation (prefix rows; theta=0 gen shape; anchor engagement) ----
    print("=== C4. Isolation (prefix rows zero gen activity; theta=0 gen row "
          "keeps legacy shape) ===")
    for name in DETERMINISM_ROWS:
        recs = load_records(V8, name)
        g1_leak = [g for g in gates_of(recs)
                   if ("cap_gen_pool" in g) or ("checked_cap_ids" in g)]
        markers = [r for r in recs if r.get("sccl_gen_probe")]
        st = (v8.get(name, {}).get("sccl", {}) or {})
        gen_made = int(st.get("gen_probes_committed", 0))
        anchor = [ui for ui in accepted_infos(recs)
                  if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        if g1_leak or markers or gen_made or anchor:
            failures.append(f"C4 isolation: {name} shows G1/anchor activity "
                            f"(g1_fields={len(g1_leak)}, markers={len(markers)}, "
                            f"gen_committed={gen_made}, anchor_updates={len(anchor)})")
        else:
            print(f"  {name:22s} isolation OK (no gen activity, no anchor)")
    recs = load_records(V8, THETA0_GEN_ROW)
    e1_leak = [g for g in gates_of(recs)
               if ("cap_rates" in g) or ("cap_retain_min" in g) or ("cap_n" in g)]
    if e1_leak:
        failures.append(f"C4: {THETA0_GEN_ROW} (theta=0) logged E1 fields on "
                        f"{len(e1_leak)} gates (legacy gate shape violated)")
    else:
        cg = cap_gates(recs)
        print(f"  {THETA0_GEN_ROW:22s} theta=0 legacy shape OK "
              f"(checked-cap gates={len(cg)}, no E1 fields)")

    print("=== C4b. Anchor engagement ===")
    acc = accepted_infos(load_records(V8, ANCHOR_ROW))
    pen_pos = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
    lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
    print(f"  {ANCHOR_ROW:22s} accepted={len(acc)} anchor_pen>0={len(pen_pos)} "
          f"lambdas={sorted(lams)}")
    if acc and len(pen_pos) != len(acc):
        failures.append(f"C4b: {ANCHOR_ROW} has {len(acc) - len(pen_pos)}/{len(acc)} "
                        "accepted updates with anchor_pen == 0 (anchor disengaged)")
    if acc and lams != {0.1}:
        failures.append(f"C4b: {ANCHOR_ROW} anchor_lambda {sorted(lams)} != [0.1]")
    for name in NONANCHOR_ROWS:
        n_leak = sum(1 for ui in accepted_infos(load_records(V8, name))
                     if float(ui.get("anchor_pen", 0.0) or 0.0) > 0)
        if n_leak:
            failures.append(f"C4b: {name} (non-anchor row) logged anchor_pen>0 on "
                            f"{n_leak} updates (anchor leaked)")

    # ---- C5. gold-free (AST audit: selfcert + manufacture path + vault) ----
    print("=== C5. Gold-free (AST audit: selfcert.py, _gen_make, vault.py) ===")
    path = os.path.join(REPO, "gcl", "selfcert.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    violations = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(tree)
                  if isinstance(n, ast.Attribute) and n.attr in GOLD_FIELDS]
    if violations:
        failures.append(f"C5 gold-free: selfcert.py touches gold fields: {violations}")
        print(f"  selfcert.py VIOLATIONS: {violations}")
    else:
        print("  gcl/selfcert.py touches no gold field (certification is spec-only)")

    # manufacture path: _gen_make in experiment.py must consume no gold label
    # field; entry_point counts as gold when sccl_derive_entry is on in the
    # run config (it must never be read — the entry is derived from the spec).
    cfg_exp = json.load(open(CONFIG)).get("experiment", {})
    derive = bool(cfg_exp.get("sccl_derive_entry", True))
    forbidden = LABEL_FIELDS | ({"entry_point"} if derive else set())
    exp_path = os.path.join(REPO, "gcl", "experiment.py")
    exp_tree = ast.parse(open(exp_path, encoding="utf-8").read())
    gen_make = next((n for n in ast.walk(exp_tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_gen_make"),
                    None)
    if gen_make is None:
        failures.append("C5: _gen_make not found in gcl/experiment.py")
    else:
        gv = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(gen_make)
              if isinstance(n, ast.Attribute) and n.attr in forbidden]
        if gv:
            failures.append(f"C5 gold-free: _gen_make touches gold fields: {gv}")
            print(f"  _gen_make VIOLATIONS: {gv}")
        else:
            print(f"  _gen_make consumes no gold field "
                  f"(forbidden set: {sorted(forbidden)})")

    # vault.py is audited in three fail-closed layers (the pre-registration
    # scopes C5 to "the manufacture AND gate paths"; a blanket name ban is
    # unsound here because _Skill stores each skill's OWN self-certified
    # tests under the attribute name `test_code` — provenance: certifier
    # self_tests, never Task gold — so the attribute NAME alone cannot
    # distinguish gold from vault-internal state):
    vault_path = os.path.join(REPO, "gcl", "vault.py")
    vtree = ast.parse(open(vault_path, encoding="utf-8").read())

    # (1) every method the SCCL pipeline calls on the vault must be Task-free:
    # no parameter named task, no Task attribute access anywhere in its body.
    SCCL_VAULT_METHODS = ["commit_certified", "commit_probe", "commit_cap_probe",
                          "retire_cap_probe", "_cap_pool_by_family",
                          "selfreplay_veto", "_too_similar_spec_exists"]
    vmethods = {n.name: n for n in ast.walk(vtree)
                if isinstance(n, ast.FunctionDef)}
    for m in SCCL_VAULT_METHODS:
        fn = vmethods.get(m)
        if fn is None:
            failures.append(f"C5: vault method {m} not found in gcl/vault.py")
            continue
        has_task_param = any(a.arg == "task" for a in fn.args.args)
        gold_attrs = [f"line {n.lineno}: .{n.attr}"
                      for n in ast.walk(fn)
                      if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "task"
                      and n.attr in GOLD_FIELDS]
        if has_task_param or gold_attrs:
            failures.append(f"C5 gold-free: vault.{m} takes a Task object "
                            f"(task param={has_task_param}, gold attrs={gold_attrs})")
    if not any(f.startswith("C5") for f in failures):
        print(f"  vault SCCL decision methods Task-free: {SCCL_VAULT_METHODS}")

    # (2) exhaustive enumeration: the ONLY raw Task-gold readers in vault.py
    # are the legacy VSR methods. Any reader outside that set — including a
    # NEW one added to an SCCL path later — fails the audit. This proves no
    # Task gold can reach any SCCL decision path, name-collisions aside.
    LEGACY_VSR_METHODS = {"commit", "violates", "choose_target", "_too_similar_exists"}
    readers: dict[str, list[str]] = {}
    for n in ast.walk(vtree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        hits = [f"line {a.lineno}: .{a.attr}" for a in ast.walk(n)
                if isinstance(a, ast.Attribute)
                and isinstance(a.value, ast.Name) and a.value.id == "task"
                and a.attr in GOLD_FIELDS]
        if hits:
            readers[n.name] = hits
    legacy_only = set(readers).issubset(LEGACY_VSR_METHODS)
    if not legacy_only:
        failures.append("C5 gold-free: Task-gold readers OUTSIDE the legacy "
                        f"VSR set: {sorted(set(readers) - LEGACY_VSR_METHODS)}")
    else:
        print(f"  Task-gold readers confined to legacy VSR methods: "
              f"{sorted(readers.keys())} (unreachable in sccl mode; layer 3 proves it)")

    # (3) run-data proof the legacy VSR paths never executed in THIS ladder:
    # no vsr-gate method, no gold target source, no decision taken by a
    # non-sccl gate on any row (frozen takes no gate decisions at all).
    for name in DETERMINISM_ROWS + GEN_ROWS:
        recs = load_records(V8, name)
        for ui in update_infos(recs):
            g = ui.get("gate") or {}
            if g.get("method") == "vsr":
                failures.append(f"C5 gold-free: {name} took a legacy VSR gate "
                                "decision (method=='vsr') in the run data")
            if ui.get("target_source") == "gold":
                failures.append(f"C5 gold-free: {name} update consumed a gold "
                                "target (target_source=='gold')")
    if not any("VSR gate decision" in f or "gold target" in f for f in failures):
        print("  run data: zero vsr-gate decisions and zero gold targets across all rows")

    if failures:
        print("\n[check] FAIL-CLOSED check(s) FAILED — aborting before decision rules:")
        for f in failures:
            print(f"  - {f}")
        out = os.path.join(V8, "verdict_check.md")
        with open(out, "w") as f:
            f.write("# SCCL v8 — FAIL-CLOSED ABORT\n\n")
            for x in failures:
                f.write(f"- {x}\n")
        print(f"[check] wrote {out}")
        sys.exit(1)

    # ---- degeneracy guard (pre-registered): <5 accepted updates ----
    degenerate = set()
    for name in GEN_ROWS:
        if name == THETA0_GEN_ROW:
            continue
        upd = v8.get(name, {}).get("report", {}).get("updates", 0)
        if int(upd) < DEGENERATE_MIN_UPDATES:
            degenerate.add(name)
            print(f"  [degeneracy] {name}: {upd} accepted updates < "
                  f"{DEGENERATE_MIN_UPDATES} -> DEGENERATE (dose finding, "
                  "recorded per pre-registration; excluded from maxima)")

    # ---- pre-registered decision rules (only if all fail-closed checks pass) ----
    print("=== Pre-registered decision rules (Branch F) ===")
    sc = v8["sccl"]
    gen = v8.get(THETA0_GEN_ROW, {})
    strict = v8.get("sccl_genprobe_strict", {})
    comp = v8.get(ANCHOR_ROW, {})
    a_gen = fam_scores(gen)["arith"] if gen else float("nan")
    a_strict = fam_scores(strict)["arith"] if strict else float("nan")
    a_comp = fam_scores(comp)["arith"] if comp else float("nan")
    f_sc = sc.get("frontier_score")
    f_comp = comp.get("frontier_score")
    u_strict = int(strict.get("report", {}).get("updates", 0)) if strict else 0
    u_comp = int(comp.get("report", {}).get("updates", 0)) if comp else 0

    h1 = a_gen > BASELINE_ARITH
    h2 = ("sccl_genprobe_strict" not in degenerate
          and a_strict >= max(BASELINE_ARITH, a_gen) and u_strict >= DEGENERATE_MIN_UPDATES)
    cells = [n for n in GEN_ROWS if n not in degenerate]
    cell_arith = {n: fam_scores(v8[n])["arith"] for n in cells}
    cell_acc = {n: v8[n]["report"]["acc"] for n in cells}
    h3_arith = bool(comp) and cells and a_comp >= max(cell_arith.values())
    h3_acc = bool(comp) and cells and \
        comp.get("report", {}).get("acc", -1) >= max(cell_acc.values())
    h3 = h3_arith and h3_acc
    bt = (ANCHOR_ROW not in degenerate and a_comp >= 0.55
          and f_comp is not None and f_sc is not None
          and f_comp >= f_sc - 0.02 and u_comp >= DEGENERATE_MIN_UPDATES)

    print(f"  H1  coverage    : arith genprobe={a_gen:.3f} > baseline "
          f"{BASELINE_ARITH} -> {'PASS' if h1 else 'FAIL'}")
    print(f"  H2  cov+sens    : arith strict={a_strict:.3f} >= max({BASELINE_ARITH}, "
          f"{a_gen:.3f}) AND updates {u_strict} >= {DEGENERATE_MIN_UPDATES} -> "
          f"{'PASS' if h2 else 'FAIL'}"
          + (" [DEGENERATE]" if "sccl_genprobe_strict" in degenerate else ""))
    if cells:
        print(f"  H3  composition : arith {a_comp:.3f} >= max({cell_arith}) AND "
              f"acc {comp.get('report', {}).get('acc', -1):.3f} >= max({cell_acc}) -> "
              f"{'PASS' if h3 else 'FAIL'}"
              + (" [degenerate rows excluded]" if degenerate else ""))
    else:
        print("  H3  composition : no non-degenerate gen rows -> FAIL")
    print(f"  BREAKTHROUGH    : arith {a_comp:.3f} >= 0.55 AND frontier {f_comp:+.3f} "
          f">= {f_sc:+.3f}-0.02 AND updates {u_comp} >= {DEGENERATE_MIN_UPDATES} -> "
          f"{'PASS' if bt else 'FAIL'}"
          + (f" [{ANCHOR_ROW} DEGENERATE]" if ANCHOR_ROW in degenerate else ""))
    if bt:
        print("  -> BREAKTHROUGH rule met: multi-seed confirmation REQUIRED "
              "(scripts/v8_multiseed.py) before any headline claim.")

    # ---- M1 mechanism telemetry (recorded, not gated) ----
    print("=== M1. Gen-witness sensitivity (mechanism telemetry, not gated) ===")
    m1_vals = {}
    for name in GEN_ROWS:
        fam_of: dict = {}
        gate_of: dict = {}
        recs = load_records(V8, name)
        for r in recs:
            ui = r.get("update_info") or {}
            v = ui.get("adapter_version")
            if v is not None and ui.get("accepted"):
                fam_of[int(v)] = r.get("family", "?")
                gate_of[int(v)] = ui.get("gate") or {}
        registry_p = os.path.join(V8, "adapters_" + name, "registry.json")
        metas = (json.load(open(registry_p)).get("metas", [])
                 if os.path.exists(registry_p) else [])
        prev_cand, n_erosion, n_gen_seen = None, 0, 0
        for m in metas:
            g = m.get("gate") or {}
            gt = g.get("gold_telemetry")
            if not gt:
                continue
            cand = float(gt["cand"])
            if prev_cand is None:
                prev_cand = float(gt["base"])
            eroding = (cand - prev_cand) < -EPS
            prev_cand = cand
            if not (eroding and int(g.get("checked_cap", 0)) > 0):
                continue
            n_erosion += 1
            gen_break = any(i.endswith(":g") for i in g.get("broke_cap", []))
            gpr = gen_probe_rate(g)
            theta = THETA_ROWS.get(name)
            gen_sub = (gpr is not None and theta is not None
                       and gpr < theta - 1e-9)
            if gen_break or gen_sub:
                n_gen_seen += 1
        m1_vals[name] = (n_gen_seen, n_erosion)
        pct = (100.0 * n_gen_seen / n_erosion) if n_erosion else None
        print(f"  {name:22s} probe-checked gold erosions={n_erosion}; with ':g' "
              f"break or gen rate<theta: {n_gen_seen}"
              + (f" ({pct:.0f}%)" if pct is not None else " (no erosions)"))
    m1_top = max((s / n for s, n in m1_vals.values() if n > 0), default=None)
    if m1_top is not None:
        print(f"  M1 verdict (target > {M1_TARGET:.0%}): max sensitivity "
              f"{m1_top:.1%} -> {'MET' if m1_top > M1_TARGET else 'BELOW TARGET'} "
              "(recorded; not gated)")
    else:
        print("  M1 verdict: no probe-checked gold erosions on any gen row "
              "(nothing to measure)")

    # ---- verdict table ----
    print("=== Verdict table ===")
    rows = []
    header = (f"| {'learner':26s} | ACC   | BWT    | Forget | Frontier | Upd  "
              f"| arith | math  | string | drift |")
    rows.append(header)
    rows.append("|" + "-" * (len(header) - 2) + "|")
    for name in DETERMINISM_ROWS + GEN_ROWS:
        L = v8.get(name)
        if L is None:
            continue
        rep, fams = L["report"], fam_scores(L)
        flag = " *" if name in degenerate else ""
        rows.append(
            f"| {name + flag:26s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
            f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
            f"| {rep['updates']:>3d}  | {fams['arith']:.3f} | {fams['math_word']:.3f} "
            f"| {fams['string']:.3f}  | {fams['drift']:.3f} |")
        print(rows[-1])

    out = os.path.join(V8, "verdict_check.md")
    with open(out, "w") as f:
        f.write("# SCCL v8 (Branch F, G1 generalization witnesses) — verdict\n\n")
        f.write(f"H1: {'PASS' if h1 else 'FAIL'} | "
                f"H2: {'PASS' if h2 else 'FAIL'} | "
                f"H3: {'PASS' if h3 else 'FAIL'} | "
                f"M1: " + (f"{m1_top:.1%} "
                           f"({'MET' if m1_top > M1_TARGET else 'BELOW TARGET'})"
                           if m1_top is not None else "n/a") + " | "
                f"BREAKTHROUGH: {'PASS' if bt else 'FAIL'}\n\n")
        f.write("All fail-closed checks (C1-C5) passed.\n")
        if degenerate:
            f.write(f"Degenerate rows (<{DEGENERATE_MIN_UPDATES} updates, dose "
                    f"finding per pre-registration): {sorted(degenerate)}\n")
        f.write("\n" + "\n".join(rows) + "\n")
    print(f"\n[check] wrote {out}")
    print("[check] all fail-closed checks passed")


if __name__ == "__main__":
    main()
