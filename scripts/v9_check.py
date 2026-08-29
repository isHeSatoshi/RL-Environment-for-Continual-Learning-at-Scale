"""Post-run verification for SCCL v9 (Branch G — runs/sccl_v9_s{42,43,44}).

Run AFTER all three v9 seeds finalize (the 3-seed ladder IS the experiment;
run_seeds.py has already written runs/sccl_v9_seeds/aggregate.json):
    python scripts/v9_check.py

Enforces the FAIL-CLOSED checks and PRE-REGISTERED decision rules committed
in RESEARCH_NOTES_v5.md "BRANCH G PRE-REGISTRATION (V9 DOSE LADDER)"
(commit ad7172d, BEFORE any v9 implementation). Any fail-closed check failing
on ANY seed => abort, do NOT interpret the decision rules.

  C1 determinism   — per seed s, the four prefix rows (frozen, sccl,
                     sccl_capprobe, sccl_capprobe_strat) must bit-reproduce
                     the matching v8 run (s42 vs runs/sccl_v8; s43/s44 vs
                     runs/sccl_v8_s{43,44}) on every computed metric,
                     frontier, and family holdout score.
  C2 engagement    — each treatment row manufactures G=2 witnesses at first
                     contact (markers; committed >= 4 across the run), ':g'
                     ids in checked_cap_ids on >= 1 gate, cap_retain_min ==
                     {0.5 | 2/3} and cap_n == 3 on every checked gate,
                     denominators in {3,5}; the final vault gen lane holds
                     >= 2 ':g' probes for every family unless the family's
                     untrained sources were exhausted (logged, not failed).
  C3 retirement    — no ':g' probe whose source task T is checked at a gate
                     during or after T's own training episode; every ':g'
                     source is a stream TRAINING task, disjoint from the
                     gold holdout task ids in eval_detail.final_heldout.
  C4 isolation     — prefix rows show ZERO gen/E1/anchor activity; BOTH
                     treatment rows log anchor_pen > 0 with lambda == 0.1
                     on every accepted update.
  C5 gold-free     — the v8 three-layer audit per seed: selfcert.py global
                     AST; _gen_make vs LABEL_FIELDS (+ entry_point under
                     sccl_derive_entry=true); vault SCCL methods
                     Task-object-free; Task-gold readers confined to the
                     legacy VSR set; zero vsr-method / gold-target records.

  H1 (plasticity restored) — per seed: every treatment row >= 8 accepted
                     updates (v8 theta=1.0 gave 14/4/6).
  H2 (dominance kept)      — per seed: every treatment row arith >=
                     sccl.arith (paired).
  H3 (breakthrough, per seed) — some treatment row arith >= 0.55 AND
                     frontier >= sccl.frontier - 0.02 AND updates >= 5.
  V9 VERDICT: BREAKTHROUGH CONFIRMED iff H3 at all 3 seeds; DOMINANCE
                     CONFIRMED iff H2 at all 3 seeds. H1 FAIL at any seed
                     => dose finding (theta axis exhausted). No interpolated
                     theta will be run.
  DEGENERACY GUARD: a treatment row with < 5 accepted updates at a seed is
                     degenerate there (excluded from that seed's H3 maxima;
                     forces H3 FAIL there).
  M1 telemetry (recorded, not gated) as in v8.

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
CONFIG = os.path.join(REPO, "configs", "sccl_v9.json")
SEEDS = [42, 43, 44]
V9_DIR = {s: os.path.join(REPO, f"runs/sccl_v9_s{s}") for s in SEEDS}
V8_REF = {42: os.path.join(REPO, "runs", "sccl_v8"),
          43: os.path.join(REPO, "runs", "sccl_v8_s43"),
          44: os.path.join(REPO, "runs", "sccl_v8_s44")}
DETERMINISM_ROWS = ["frozen", "sccl", "sccl_capprobe", "sccl_capprobe_strat"]
TREATMENT_ROWS = ["sccl_gen2_half", "sccl_gen2_majority"]
THETA_ROWS = {"sccl_gen2_half": 0.5, "sccl_gen2_majority": 2 / 3}
ALL_ROWS = DETERMINISM_ROWS + TREATMENT_ROWS
REPORT_KEYS = ["acc", "bwt", "fwt", "forgetting", "auc", "stability", "updates"]
FAMILIES = ["arith", "math_word", "string", "drift"]
LABEL_FIELDS = {"test_code", "reference_answer", "test_list", "reference"}
GOLD_FIELDS = LABEL_FIELDS | {"entry_point"}
CAP_N = 3
CAP_MARGIN = 2
G_LANE = 2
MIN_COMMITTED = 4
DEGENERATE_MIN_UPDATES = 5
PLASTICITY_MIN_UPDATES = 8
BT_ARITH = 0.55
BT_FRONTIER_EPS = 0.02
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
    rates = g.get("cap_rates") or {}
    gen = {k: v for k, v in rates.items() if k.endswith(":g")}
    if not gen:
        return None
    passes = sum(int(v.get("passes", 0)) for v in gen.values())
    n = sum(int(v.get("n", 0)) for v in gen.values())
    return passes / n if n > 0 else None


def per_seed_checks(seed: int, failures: list) -> dict:
    """All fail-closed checks for one seed. Returns the learners dict for
    the decision-rule layer."""
    run = V9_DIR[seed]
    ref = V8_REF[seed]
    v9 = load_metrics(run)
    refm = load_metrics(ref)
    print(f"\n########## seed {seed} ({run}) vs reference {ref} ##########")
    HOLD = holdout_ids(v9)

    # ---- C1. determinism vs the matching v8 run (prefix rows) ----
    print(f"=== C1. seed {seed}: prefix bit-identity vs {ref} ===")
    for name in DETERMINISM_ROWS:
        a, b = refm.get(name), v9.get(name)
        if a is None or b is None:
            failures.append(f"C1[s{seed}]: learner {name} missing from a run")
            continue
        diffs = []
        for k in REPORT_KEYS:
            va, vb = a["report"].get(k), b["report"].get(k)
            if va != vb:
                diffs.append(f"{k}: {va} -> {vb}")
        if a.get("frontier_score") != b.get("frontier_score"):
            diffs.append(f"frontier: {a.get('frontier_score')} -> "
                         f"{b.get('frontier_score')}")
        for fam in FAMILIES:
            sa, sb = fam_scores(a).get(fam), fam_scores(b).get(fam)
            if sa != sb:
                diffs.append(f"hold[{fam}]: {sa} -> {sb}")
        status = "BIT-IDENTICAL" if not diffs else "MISMATCH"
        print(f"  {name:22s} {status}" + ("" if not diffs else f"  {diffs}"))
        if diffs:
            failures.append(f"C1[s{seed}]: {name} differs from v8: {diffs}")

    # ---- C2. G2 engagement ----
    print(f"=== C2. seed {seed}: G=2 engagement (manufacture, lanes, dose) ===")
    lane_report = {}
    for name in TREATMENT_ROWS:
        recs = load_records(run, name)
        markers = [r for r in recs if r.get("sccl_gen_probe")]
        # src is a LIST of live witness sources since the G-lane change
        first_contact_counts = [len(m["sccl_gen_probe"].get("src") or [])
                                for m in markers]
        cg = cap_gates(recs)
        gen_gates = [g for g in cg
                     if any(i.endswith(":g") for i in g.get("checked_cap_ids", []))]
        max_gen_ids = max((sum(1 for i in g.get("checked_cap_ids", [])
                               if i.endswith(":g")) for g in cg), default=0)
        st = (v9.get(name, {}).get("sccl", {}) or {})
        made = int(st.get("gen_probes_made", 0))
        committed = int(st.get("gen_probes_committed", 0))
        retired = int(st.get("gen_probes_retired", 0))
        # final vault gen lane per family
        lane = {}
        vp = os.path.join(run, f"vault_{name}", "skills.json")
        if os.path.exists(vp):
            for r in json.load(open(vp)):
                if r.get("kind") == "cap_probe" and r["task_id"].endswith(":g"):
                    lane.setdefault(r.get("family", "?"), set()).add(r["task_id"])
        lane_n = {f: len(s) for f, s in lane.items()}
        lane_report[name] = lane_n
        print(f"  {name:22s} markers={len(markers)} first-contact lane sizes="
              f"{first_contact_counts} committed={committed} retired={retired} "
              f"checked-cap gates={len(cg)} with-':g'={len(gen_gates)} "
              f"max ':g' in one gate={max_gen_ids} final vault lane={lane_n}")
        if not markers:
            failures.append(f"C2[s{seed}]: {name} manufactured no gen probe "
                            "(no sccl_gen_probe marker)")
        if committed < MIN_COMMITTED:
            failures.append(f"C2[s{seed}]: {name} committed {committed} < "
                            f"{MIN_COMMITTED} gen witnesses (lane may be "
                            "under-filled)")
        if not gen_gates:
            failures.append(f"C2[s{seed}]: {name} never checked a ':g' probe")
        theta = THETA_ROWS[name]
        bad_theta = [g for g in cg
                     if abs(float(g.get("cap_retain_min", -1.0)) - theta) > 1e-9]
        bad_n = [g for g in cg if int(g.get("cap_n", -1)) != CAP_N]
        holes = [g for g in cg if not g.get("cap_rates")]
        bad_pool = [g for g in cg if int(g.get("cap_gen_pool", -1)) != G_LANE]
        denoms = sorted({int(r["n"]) for g in cg
                         for r in (g.get("cap_rates") or {}).values()})
        legal = {CAP_N, CAP_N + CAP_MARGIN}
        bad_denoms = [d for d in denoms if d not in legal]
        gen_rate_ids = [tid for g in cg for tid in (g.get("cap_rates") or {})
                        if tid.endswith(":g")]
        print(f"  {name:22s} theta_mismatched={len(bad_theta)} "
              f"cap_n_mismatched={len(bad_n)} gen_pool_mismatched={len(bad_pool)} "
              f"rate_holes={len(holes)} denominators={denoms} "
              f"cap_rates ':g' ids={len(gen_rate_ids)}")
        if bad_theta:
            failures.append(f"C2[s{seed}]: {name} logged cap_retain_min != "
                            f"{theta} on {len(bad_theta)} gates")
        if bad_n:
            failures.append(f"C2[s{seed}]: {name} logged cap_n != {CAP_N} on "
                            f"{len(bad_n)} gates")
        if bad_pool:
            failures.append(f"C2[s{seed}]: {name} logged cap_gen_pool != "
                            f"{G_LANE} on {len(bad_pool)} gates")
        if holes:
            failures.append(f"C2[s{seed}]: {name} has {len(holes)} checked-cap "
                            "gates with EMPTY cap_rates")
        if bad_denoms:
            failures.append(f"C2[s{seed}]: {name} denominators {bad_denoms} "
                            f"outside {sorted(legal)}")
        if gen_gates and not gen_rate_ids:
            failures.append(f"C2[s{seed}]: {name} checked ':g' probes but "
                            "cap_rates never covered a ':g' id")
        short = {f: n for f, n in lane_n.items() if n < G_LANE}
        if short:
            print(f"  [lane note] {name}: families with < {G_LANE} live gen "
                  f"probes at run end: {short} (exhausted untrained sources "
                  "are a logged dose note, not a failure)")

    # ---- C3. retirement + anti-contamination ----
    print(f"=== C3. seed {seed}: retirement/anti-contamination ===")
    for name in TREATMENT_ROWS:
        recs = load_records(run, name)
        trained, bad_retire, bad_hold, checked_srcs = set(), [], [], set()
        for r in recs:
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
            # manufacture marker sources are a LIST (G-lane)
            for msrc in (r.get("sccl_gen_probe") or {}).get("src", []) or []:
                if msrc in HOLD:
                    bad_hold.append(msrc)
                if msrc in trained:
                    bad_retire.append((r.get("task_id"), msrc + ":g (mfd)"))
        print(f"  {name:22s} distinct ':g' sources checked={len(checked_srcs)} "
              f"retirement violations={len(bad_retire)} "
              f"holdout-sourced={len(set(bad_hold))}")
        if bad_retire:
            failures.append(f"C3[s{seed}] retirement: {name} checked a gen "
                            f"probe whose source was already trained: "
                            f"{bad_retire[:3]}")
        if bad_hold:
            failures.append(f"C3[s{seed}] contamination: {name} gen probes "
                            f"sourced from GOLD HOLDOUT tasks: "
                            f"{sorted(set(bad_hold))}")

    # ---- C4. isolation + anchor engagement ----
    print(f"=== C4. seed {seed}: isolation + anchor engagement ===")
    for name in DETERMINISM_ROWS:
        recs = load_records(run, name)
        leak = [g for g in gates_of(recs)
                if ("cap_gen_pool" in g) or ("checked_cap_ids" in g)]
        markers = [r for r in recs if r.get("sccl_gen_probe")]
        st = (v9.get(name, {}).get("sccl", {}) or {})
        gen_made = int(st.get("gen_probes_committed", 0))
        anchor = [ui for ui in accepted_infos(recs)
                  if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        if leak or markers or gen_made or anchor:
            failures.append(f"C4[s{seed}] isolation: {name} shows gen/anchor "
                            f"activity (fields={len(leak)}, markers="
                            f"{len(markers)}, gen_committed={gen_made}, "
                            f"anchor_updates={len(anchor)})")
        else:
            print(f"  {name:22s} isolation OK (no gen activity, no anchor)")
    for name in TREATMENT_ROWS:
        acc = accepted_infos(load_records(run, name))
        pen_pos = [ui for ui in acc if float(ui.get("anchor_pen", 0.0) or 0.0) > 0]
        lams = {round(float(ui.get("anchor_lambda", 0.0)), 6) for ui in acc}
        print(f"  {name:22s} accepted={len(acc)} anchor_pen>0={len(pen_pos)} "
              f"lambdas={sorted(lams)}")
        if acc and len(pen_pos) != len(acc):
            failures.append(f"C4b[s{seed}]: {name} has "
                            f"{len(acc) - len(pen_pos)}/{len(acc)} accepted "
                            "updates with anchor_pen == 0")
        if acc and lams != {0.1}:
            failures.append(f"C4b[s{seed}]: {name} lambdas {sorted(lams)} "
                            "!= [0.1]")

    # ---- C5. gold-free (3-layer audit; static parts are seed-independent
    #      but the run-data layer is per seed) ----
    print(f"=== C5. seed {seed}: gold-free ===")
    path = os.path.join(REPO, "gcl", "selfcert.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    viol = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr in GOLD_FIELDS]
    if viol:
        failures.append(f"C5[s{seed}] gold-free: selfcert.py touches gold "
                        f"fields: {viol}")
    else:
        print("  selfcert.py touches no gold field (certification is spec-only)")
    cfg_exp = json.load(open(CONFIG)).get("experiment", {})
    derive = bool(cfg_exp.get("sccl_derive_entry", True))
    forbidden = LABEL_FIELDS | ({"entry_point"} if derive else set())
    exp_path = os.path.join(REPO, "gcl", "experiment.py")
    exp_tree = ast.parse(open(exp_path, encoding="utf-8").read())
    gen_make = next((n for n in ast.walk(exp_tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "_gen_make"),
                    None)
    if gen_make is None:
        failures.append(f"C5[s{seed}]: _gen_make not found in experiment.py")
    else:
        gv = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(gen_make)
              if isinstance(n, ast.Attribute) and n.attr in forbidden]
        if gv:
            failures.append(f"C5[s{seed}] gold-free: _gen_make touches gold "
                            f"fields: {gv}")
        else:
            print(f"  _gen_make consumes no gold field (forbidden: "
                  f"{sorted(forbidden)})")
    vault_path = os.path.join(REPO, "gcl", "vault.py")
    vtree = ast.parse(open(vault_path, encoding="utf-8").read())
    vmethods = {n.name: n for n in ast.walk(vtree)
                if isinstance(n, ast.FunctionDef)}
    SCCL_VAULT_METHODS = ["commit_certified", "commit_probe", "commit_cap_probe",
                          "retire_cap_probe", "_cap_pool_by_family",
                          "selfreplay_veto", "_too_similar_spec_exists"]
    for m in SCCL_VAULT_METHODS:
        fn = vmethods.get(m)
        if fn is None:
            failures.append(f"C5[s{seed}]: vault method {m} not found")
            continue
        has_task = any(a.arg == "task" for a in fn.args.args)
        gold_attrs = [f"line {n.lineno}: .{n.attr}" for n in ast.walk(fn)
                      if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "task"
                      and n.attr in GOLD_FIELDS]
        if has_task or gold_attrs:
            failures.append(f"C5[s{seed}] gold-free: vault.{m} takes/reads a "
                            f"Task object ({has_task}, {gold_attrs})")
    LEGACY_VSR = {"commit", "violates", "choose_target", "_too_similar_exists"}
    readers = {}
    for n in ast.walk(vtree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        hits = [f"line {a.lineno}: .{a.attr}" for a in ast.walk(n)
                if isinstance(a, ast.Attribute)
                and isinstance(a.value, ast.Name) and a.value.id == "task"
                and a.attr in GOLD_FIELDS]
        if hits:
            readers[n.name] = hits
    if not set(readers).issubset(LEGACY_VSR):
        failures.append(f"C5[s{seed}] gold-free: Task-gold readers OUTSIDE the "
                        f"legacy VSR set: {sorted(set(readers) - LEGACY_VSR)}")
    else:
        print(f"  Task-gold readers confined to legacy VSR methods: "
              f"{sorted(readers.keys())}")
    for name in ALL_ROWS:
        for ui in update_infos(load_records(run, name)):
            g = ui.get("gate") or {}
            if g.get("method") == "vsr":
                failures.append(f"C5[s{seed}] gold-free: {name} took a legacy "
                                "VSR gate decision")
            if ui.get("target_source") == "gold":
                failures.append(f"C5[s{seed}] gold-free: {name} consumed a "
                                "gold target")
    if not any(f.startswith(f"C5[s{seed}]") and ("VSR gate" in f or "gold target" in f)
               for f in failures):
        print("  run data: zero vsr-gate decisions and zero gold targets")
    return v9


def m1_of(run: str, name: str) -> tuple[int, int]:
    """(gen_seen, probe_checked_erosions) from registry metas."""
    fam_of: dict = {}
    gate_of: dict = {}
    for r in load_records(run, name):
        ui = r.get("update_info") or {}
        v = ui.get("adapter_version")
        if v is not None and ui.get("accepted"):
            fam_of[int(v)] = r.get("family", "?")
            gate_of[int(v)] = ui.get("gate") or {}
    registry_p = os.path.join(run, "adapters_" + name, "registry.json")
    metas = (json.load(open(registry_p)).get("metas", [])
             if os.path.exists(registry_p) else [])
    prev_cand, n_erosion, n_seen = None, 0, 0
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
            n_seen += 1
    return n_seen, n_erosion


def main() -> None:
    failures: list[str] = []
    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        per_seed[seed] = per_seed_checks(seed, failures)

    if failures:
        print("\n[check] FAIL-CLOSED check(s) FAILED — aborting before "
              "decision rules:")
        for f in failures:
            print(f"  - {f}")
        out = os.path.join(REPO, "runs", "sccl_v9_seeds", "verdict_check.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write("# SCCL v9 — FAIL-CLOSED ABORT\n\n")
            for x in failures:
                f.write(f"- {x}\n")
        print(f"[check] wrote {out}")
        sys.exit(1)

    # ---- degeneracy guard per seed ----
    degenerate: dict[int, set] = {}
    print("\n=== Degeneracy guard (<5 accepted updates at a seed) ===")
    for seed in SEEDS:
        degenerate[seed] = set()
        for name in TREATMENT_ROWS:
            upd = int(per_seed[seed].get(name, {})
                      .get("report", {}).get("updates", 0))
            if upd < DEGENERATE_MIN_UPDATES:
                degenerate[seed].add(name)
                print(f"  s{seed} {name}: {upd} updates -> DEGENERATE at this "
                      "seed (excluded from H3 maxima here; forces H3 FAIL)")

    # ---- pre-registered decision rules, per seed then across seeds ----
    print("=== Pre-registered decision rules (Branch G) ===")
    h1, h2, h3 = {}, {}, {}
    for seed in SEEDS:
        v9 = per_seed[seed]
        sc_ar = fam_scores(v9["sccl"])["arith"]
        sc_fr = v9["sccl"].get("frontier_score")
        rowstats = {}
        for name in TREATMENT_ROWS:
            L = v9.get(name, {})
            ar = fam_scores(L)["arith"]
            fr = L.get("frontier_score")
            upd = int(L.get("report", {}).get("updates", 0))
            rowstats[name] = (ar, fr, upd)
        h1[seed] = all(u >= PLASTICITY_MIN_UPDATES
                       for _, _, u in rowstats.values())
        h2[seed] = all(ar >= sc_ar for ar, _, _ in rowstats.values())
        cands = [n for n in TREATMENT_ROWS if n not in degenerate[seed]]
        h3[seed] = any(
            ar >= BT_ARITH and fr is not None and sc_fr is not None
            and fr >= sc_fr - BT_FRONTIER_EPS and u >= DEGENERATE_MIN_UPDATES
            for n in cands for (ar, fr, u) in [rowstats[n]])
        print(f"  s{seed}: sccl arith={sc_ar:.3f} frontier={sc_fr:+.3f}")
        for name, (ar, fr, upd) in rowstats.items():
            print(f"    {name:22s} arith={ar:.3f} frontier={fr:+.3f} "
                  f"updates={upd}"
                  + (" (DEGENERATE)" if name in degenerate[seed] else ""))
        print(f"    H1 plasticity(>={PLASTICITY_MIN_UPDATES}): "
              f"{'PASS' if h1[seed] else 'FAIL'} | "
              f"H2 dominance(>=sccl): {'PASS' if h2[seed] else 'FAIL'} | "
              f"H3 breakthrough: {'PASS' if h3[seed] else 'FAIL'}")

    bt_confirmed = all(h3[s] for s in SEEDS)
    dom_confirmed = all(h2[s] for s in SEEDS)
    plasticity = all(h1[s] for s in SEEDS)
    print(f"\n  V9 VERDICT: BREAKTHROUGH "
          f"{'CONFIRMED' if bt_confirmed else 'NOT CONFIRMED'} | DOMINANCE "
          f"{'CONFIRMED' if dom_confirmed else 'NOT CONFIRMED'} | plasticity "
          f"{'restored' if plasticity else 'STILL FRAGILE (dose finding)'}")
    if not plasticity:
        print("  -> theta axis exhausted per pre-registration; next branch "
              "pre-registers a witness-free drift bound. NO interpolated "
              "theta will be run.")
    elif not bt_confirmed:
        print("  -> theta axis exhausted (H1 passed, H3 failed); same next "
              "branch. NO interpolated theta will be run.")

    # ---- M1 telemetry (recorded, not gated) ----
    print("\n=== M1. Gen-witness sensitivity (mechanism telemetry, not gated) ===")
    m1_lines = []
    for seed in SEEDS:
        for name in TREATMENT_ROWS:
            seen, eros = m1_of(V9_DIR[seed], name)
            pct = (100.0 * seen / eros) if eros else None
            line = (f"  s{seed} {name:22s} probe-checked erosions={eros}; "
                    f"':g'-seen={seen}"
                    + (f" ({pct:.0f}%)" if pct is not None else " (none)"))
            print(line)
            m1_lines.append(line)

    # ---- verdict table per seed ----
    print("\n=== Verdict tables ===")
    tables = []
    for seed in SEEDS:
        v9 = per_seed[seed]
        header = (f"| seed {seed} {'learner':20s} | ACC   | BWT    | Forget "
                  f"| Frontier | Upd  | arith | math  | string | drift |")
        rows = [header, "|" + "-" * (len(header) - 2) + "|"]
        for name in ALL_ROWS:
            L = v9.get(name)
            if L is None:
                continue
            rep, fams = L["report"], fam_scores(L)
            flag = " *" if name in degenerate[seed] else ""
            rows.append(
                f"| {name + flag:26s} | {rep['acc']:.3f} | {rep['bwt']:+.3f} "
                f"| {rep['forgetting']:.3f}  | {L.get('frontier_score', 0):+.3f}   "
                f"| {rep['updates']:>3d}  | {fams['arith']:.3f} "
                f"| {fams['math_word']:.3f} | {fams['string']:.3f}  "
                f"| {fams['drift']:.3f} |")
        tables.append("\n".join(rows))
        print("\n".join(rows) + "\n")

    out = os.path.join(REPO, "runs", "sccl_v9_seeds", "verdict_check.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("# SCCL v9 (Branch G dose ladder) — 3-seed verdict\n\n")
        f.write(f"Per-seed: H1 "
                + " | ".join(f"s{s}:{'PASS' if h1[s] else 'FAIL'}" for s in SEEDS)
                + "\n")
        f.write(f"Per-seed: H2 "
                + " | ".join(f"s{s}:{'PASS' if h2[s] else 'FAIL'}" for s in SEEDS)
                + "\n")
        f.write(f"Per-seed: H3 "
                + " | ".join(f"s{s}:{'PASS' if h3[s] else 'FAIL'}" for s in SEEDS)
                + "\n\n")
        f.write(f"BREAKTHROUGH: {'CONFIRMED' if bt_confirmed else 'NOT CONFIRMED'} "
                f"(H3 at all 3 seeds)\n")
        f.write(f"DOMINANCE: {'CONFIRMED' if dom_confirmed else 'NOT CONFIRMED'} "
                f"(H2 at all 3 seeds)\n")
        f.write(f"PLASTICITY: {'restored (H1 all seeds)' if plasticity else 'STILL FRAGILE — dose finding; theta axis exhausted'}\n")
        f.write("All fail-closed checks (C1-C5) passed on all seeds.\n")
        for seed in SEEDS:
            if degenerate[seed]:
                f.write(f"s{seed} degenerate rows: {sorted(degenerate[seed])}\n")
        f.write("\n")
        f.write("\n\n".join(tables) + "\n\n")
        f.write("\n".join(m1_lines) + "\n")
    print(f"[check] wrote {out}")
    print("[check] all fail-closed checks passed on all seeds")


if __name__ == "__main__":
    main()
