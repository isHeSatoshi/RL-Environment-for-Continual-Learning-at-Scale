"""Post-hoc gold telemetry for the v8 ladder (Branch F, G1 generalization witnesses).

MEASUREMENT ONLY. Gold scores were logged by the gate at run time but NEVER
entered any accept/reject decision; this script tabulates them after the fact,
same protocol as v5b/v6/v7. It extends scripts/v7_telemetry.py with the G1
evidence layer:

  1. M1 GEN-WITNESS SENSITIVITY (the pre-registered mechanism metric). v7's
     verdict found 0% sensitivity: every probe-checked gold erosion passed its
     certified-skill probes at pooled rate 1.0, because those probes witness
     REPRODUCTION of trained skills while the erosion lives on the
     GENERALIZATION axis. G1 adds witnesses manufactured from UNTRAINED tasks
     (':g' ids). The central telemetry question: at the gates that ACCEPTED an
     update which eroded the arith gold holdout, did a ':g' witness break or
     show pooled rate < theta? M1 = that fraction, per row; target > 50%.
  2. GEN-WITNESS QUALITY. Gen probes are harder than certified variants
     (untrained tasks, temp-0.7 regeneration): the per-row distribution of
     ':g' pooled rates vs ':c' pooled rates quantifies the witness difficulty
     and calibrates the theta dose for v9.
  3. MANUFACTURE/RETIREMENT LEDGER. Per row: gen probes attempted/committed/
     retired (metrics sccl counters) + the first-contact manufacture events
     from trajectory markers — the coverage ledger the pre-registration logs
     when a family cannot certify a witness.
  4. Everything from v7: erosion timelines, E1 rates, anchor engagement.

Usage: python scripts/v8_telemetry.py [--run runs/sccl_v8]
Writes <run>/telemetry_v8.json and prints a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


LEARNERS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat",
            "sccl_genprobe", "sccl_genprobe_strict", "sccl_genprobe_strict_anchor"]
THETA_ROWS = {"sccl_genprobe_strict": 1.0, "sccl_genprobe_strict_anchor": 1.0}
GEN_ROWS = {"sccl_genprobe", "sccl_genprobe_strict", "sccl_genprobe_strict_anchor"}
ANCHOR_ROWS = {"sccl_genprobe_strict_anchor"}
EPS = 0.05


def load_updates(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, f"trajectories_{name}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_registry(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, "adapters_" + name, "registry.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f).get("metas", [])


def cap_source(task_id: str) -> str:
    return task_id.rsplit(":c", 1)[0]


def vault_pool(run_dir: str, name: str) -> dict:
    """family -> {'certified': [source skills], 'gen': [':g' ids]}."""
    p = os.path.join(run_dir, f"vault_{name}", "skills.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        recs = json.load(f)
    out: dict = {}
    for r in recs:
        if r.get("kind") != "cap_probe":
            continue
        fam = r.get("family", "?")
        slot = out.setdefault(fam, {"certified": set(), "gen": set()})
        if r["task_id"].endswith(":g"):
            slot["gen"].add(r["task_id"])
        else:
            slot["certified"].add(cap_source(r["task_id"]))
    return {f: {"certified": sorted(s["certified"]), "gen": sorted(s["gen"])}
            for f, s in out.items()}


def pooled_rate(cap_rates: dict, suffix: str | None = None) -> float | None:
    """sum(passes)/sum(n) over per-probe entries (optionally ':c'/':g' only)."""
    if not cap_rates:
        return None
    if suffix is not None:
        cap_rates = {k: v for k, v in cap_rates.items()
                     if (k.endswith(":g") if suffix == ":g"
                         else not k.endswith(":g"))}
    if not cap_rates:
        return None
    passes = sum(int(v.get("passes", 0)) for v in cap_rates.values())
    n = sum(int(v.get("n", 0)) for v in cap_rates.values())
    return passes / n if n > 0 else None


def gen_ledger(run_dir: str, name: str, metrics: dict) -> dict:
    """Manufacture/retirement ledger from metrics counters + markers."""
    st = (metrics.get(name, {}).get("sccl", {}) or {})
    markers = []
    for r in load_updates(run_dir, name):
        m = r.get("sccl_gen_probe")
        if m:
            markers.append({"family": m.get("first_contact"),
                            "src": m.get("src"),
                            "at_task": r.get("task_id")})
    return {
        "gen_probes_attempted": int(st.get("gen_probes_made", 0)),
        "gen_probes_committed": int(st.get("gen_probes_committed", 0)),
        "gen_probes_retired": int(st.get("gen_probes_retired", 0)),
        "first_contact_markers": markers,
    }


def gate_stats(run_dir: str, name: str) -> dict:
    """Gate-level engagement from trajectory records (all updates, kept+vetoes)."""
    checked_hist: Counter = Counter()
    rate_hist_c: Counter = Counter()
    rate_hist_g: Counter = Counter()
    vetoed: list = []
    gen_rate_of_kept: list = []
    n_gates = 0
    n_with_ids = 0
    n_gen_checked = 0
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        g = ui.get("gate") or {}
        if "checked_cap" not in g:
            continue
        n_gates += 1
        checked_hist[int(g.get("checked_cap", 0))] += 1
        ids = g.get("checked_cap_ids")
        if ids is not None:
            n_with_ids += 1
            if any(i.endswith(":g") for i in ids):
                n_gen_checked += 1
        rates = g.get("cap_rates") or {}
        pr_c = pooled_rate(rates, ":c")
        pr_g = pooled_rate(rates, ":g")
        if pr_c is not None:
            rate_hist_c[round(pr_c * 3) / 3] += 1
        if pr_g is not None:
            rate_hist_g[round(pr_g * 3) / 3] += 1
        if g.get("broke_cap"):
            vetoed.append({
                "family": rec.get("family", "?"),
                "task_id": rec.get("task_id", "?"),
                "pooled_rate": pooled_rate(rates),
                "gen_rate": pr_g,
                "broke_cap": list(g.get("broke_cap", [])),
                "broke_gen": [i for i in g.get("broke_cap", [])
                              if i.endswith(":g")],
            })
        elif ui.get("accepted") and pr_g is not None:
            gen_rate_of_kept.append(pr_g)
    return {
        "gates_with_cap_records": n_gates,
        "gates_with_checked_cap_ids": n_with_ids,
        "gates_checking_gen_probe": n_gen_checked,
        "checked_cap_histogram": dict(sorted(checked_hist.items())),
        "cert_rate_histogram_thirds": dict(sorted(rate_hist_c.items())),
        "gen_rate_histogram_thirds": dict(sorted(rate_hist_g.items())),
        "kept_accepted_gen_rates_min": min(gen_rate_of_kept, default=None),
        "kept_accepted_gen_rates_mean": (sum(gen_rate_of_kept) / len(gen_rate_of_kept)
                                         if gen_rate_of_kept else None),
        "cap_vetoes": len(vetoed),
        "cap_vetoes_by_gen": sum(1 for v in vetoed if v["broke_gen"]),
        "cap_veto_events": vetoed,
    }


def analyze(run_dir: str, name: str) -> dict:
    # adapter_version -> (family, task_id, rates) from accepted updates
    fam_of: dict = {}
    anchor_pen_of: dict = {}
    anchor_lam_of: dict = {}
    gate_of: dict = {}
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        v = ui.get("adapter_version")
        if v is not None and ui.get("accepted"):
            fam_of[int(v)] = (rec.get("family", "?"), rec.get("task_id", "?"))
            anchor_pen_of[int(v)] = float(ui.get("anchor_pen", 0.0) or 0.0)
            anchor_lam_of[int(v)] = float(ui.get("anchor_lambda", 0.0) or 0.0)
            gate_of[int(v)] = ui.get("gate") or {}

    rows = []
    prev_cand = None
    for m in load_registry(run_dir, name):
        g = m.get("gate") or {}
        gt = g.get("gold_telemetry")
        if not gt:
            continue
        fam, task = fam_of.get(int(m["version"]), ("?", "?"))
        base = float(gt["base"])
        cand = float(gt["cand"])
        if prev_cand is None:
            prev_cand = base
        rates = g.get("cap_rates") or {}
        rows.append({
            "version": int(m["version"]),
            "family_phase": fam,
            "task_id": task,
            "checked_cap": int(g.get("checked_cap", 0)),
            "checked_gen": [i for i in g.get("checked_cap_ids", [])
                            if i.endswith(":g")],
            "broke_cap": list(g.get("broke_cap", [])),
            "broke_gen": [i for i in g.get("broke_cap", []) if i.endswith(":g")],
            "pooled_rate": pooled_rate(rates),
            "gen_rate": pooled_rate(rates, ":g"),
            "cert_rate": pooled_rate(rates, ":c"),
            "anchor_lambda": anchor_lam_of.get(int(m["version"]), 0.0),
            "anchor_pen": anchor_pen_of.get(int(m["version"]), 0.0),
            "gold_base": base,
            "gold_prev_cand": prev_cand,
            "gold_cand": cand,
            "gold_delta_vs_frozen": cand - base,
            "gold_delta_vs_prev": cand - prev_cand,
        })
        prev_cand = cand

    erosion = [r for r in rows if r["gold_delta_vs_prev"] < -EPS]
    probe_checked = [r for r in erosion if r["checked_cap"] > 0]
    probe_passed = [r for r in probe_checked if not r["broke_cap"]]
    # M1: among probe-checked gold EROSIONS, how many did a GEN witness see —
    # either broke it outright, or (theta rows) showed gen pooled rate < theta?
    theta = THETA_ROWS.get(name)
    gen_seen = []
    for r in probe_checked:
        broke = bool(r["broke_gen"])
        sub = (r["gen_rate"] is not None and theta is not None
               and r["gen_rate"] < theta - 1e-9)
        if broke or sub:
            gen_seen.append(r)
    below_frozen = [r for r in rows if r["gold_delta_vs_frozen"] < -EPS]
    first_below = next((r for r in rows if r["gold_delta_vs_frozen"] < -EPS), None)
    worst = min(rows, key=lambda r: r["gold_delta_vs_prev"]) if rows else None
    return {
        "learner": name,
        "accepted_updates_with_gold_telemetry": len(rows),
        "gold_erosion_updates": len(erosion),
        "gold_erosion_rate": round(len(erosion) / max(1, len(rows)), 3),
        "probe_checked_erosions": len(probe_checked),
        "probe_PASSED_despite_erosion": len(probe_passed),
        "insensitivity_rate": (round(len(probe_passed) / len(probe_checked), 3)
                               if probe_checked else None),
        "m1_gen_witness_seen": len(gen_seen),
        "m1_gen_witness_sensitivity": (round(len(gen_seen) / len(probe_checked), 3)
                                       if probe_checked else None),
        "m1_gen_seen_versions": [r["version"] for r in gen_seen],
        "updates_leaving_arith_below_frozen": len(below_frozen),
        "first_update_leaving_below_frozen": first_below,
        "worst_single_update": worst,
        "final_gold_cand": rows[-1]["gold_cand"] if rows else None,
        "anchor_pen_max": max((r["anchor_pen"] for r in rows), default=0.0),
        "anchor_pen_mean": (sum(r["anchor_pen"] for r in rows) / len(rows)
                            if rows else 0.0),
        "timeline": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sccl_v8")
    args = ap.parse_args()

    metrics_p = os.path.join(args.run, "metrics.json")
    metrics = json.load(open(metrics_p))["learners"] if os.path.exists(metrics_p) else {}

    out: dict = {}
    print(f"## Post-hoc gold telemetry — v8 Branch F ladder ({args.run})\n")
    print("Gold scores were logged by the gate at run time but NEVER used in any\n"
          "accept/reject decision. `gen_rate` = pooled pass rate over the ':g'\n"
          "(generalization-witness) cap_rates entries; `cert_rate` = same over\n"
          "the ':c' (certified-skill) entries. M1 = fraction of probe-checked\n"
          "gold erosions a gen witness saw (broke, or gen rate < theta).\n")

    for name in LEARNERS:
        a = analyze(args.run, name)
        a["gate_stats"] = gate_stats(args.run, name)
        a["final_cap_pool"] = vault_pool(args.run, name)
        if name in GEN_ROWS:
            a["gen_ledger"] = gen_ledger(args.run, name, metrics)
        out[name] = a

        if not a["timeline"]:
            print(f"### {name}: no gold telemetry rows\n")
            continue
        tag = []
        if name in GEN_ROWS:
            tag.append("GEN G=1")
        if name in THETA_ROWS:
            tag.append(f"E1 θ={THETA_ROWS[name]:.3f}")
        if name in ANCHOR_ROWS:
            tag.append("ANCHOR λ=0.1")
        hdr = f"### {name}" + (f"  ({', '.join(tag)})" if tag else "")
        print(hdr)
        print(f"- accepted updates w/ telemetry: {a['accepted_updates_with_gold_telemetry']}")
        print(f"- gold-erosion updates (cand < prev - {EPS}): "
              f"**{a['gold_erosion_updates']}** ({a['gold_erosion_rate']:.1%})")
        if a["probe_checked_erosions"]:
            print(f"- probe-checked erosions: {a['probe_checked_erosions']}; probe "
                  f"**PASSED** despite erosion: **{a['probe_PASSED_despite_erosion']}** "
                  f"(binary insensitivity {a['insensitivity_rate']:.1%})")
            s = a["m1_gen_witness_sensitivity"]
            print(f"- **M1 gen-witness sensitivity: "
                  f"{a['m1_gen_witness_seen']}/{a['probe_checked_erosions']}"
                  + (f" ({s:.1%})" if s is not None else "")
                  + "**  [target > 50%]")
        else:
            print("- no probe-checked erosions (probes off or none checked)")
        gl = a.get("gen_ledger")
        if gl:
            fams = sorted({m["family"] for m in gl["first_contact_markers"]
                           if m.get("family")})
            print(f"- gen ledger: attempted={gl['gen_probes_attempted']} "
                  f"committed={gl['gen_probes_committed']} "
                  f"retired={gl['gen_probes_retired']}; first-contact families: "
                  f"{fams if fams else 'NONE'}")
        gs = a["gate_stats"]
        print(f"- cap gates: {gs['gates_with_cap_records']} (with checked_cap_ids: "
              f"{gs['gates_with_checked_cap_ids']}; checking a ':g' probe: "
              f"{gs['gates_checking_gen_probe']}); cap-vetoes {gs['cap_vetoes']} "
              f"(by a ':g' witness: {gs['cap_vetoes_by_gen']})")
        if gs["gen_rate_histogram_thirds"]:
            print(f"- GEN-rate histogram (thirds): {gs['gen_rate_histogram_thirds']}; "
                  f"kept+accepted gen-rate min={gs['kept_accepted_gen_rates_min']}, "
                  f"mean={gs['kept_accepted_gen_rates_mean']}")
        if gs["cert_rate_histogram_thirds"]:
            print(f"- cert-rate histogram (thirds): {gs['cert_rate_histogram_thirds']}")
        pool = a["final_cap_pool"]
        if pool:
            pool_desc = "; ".join(
                f"{f}:{len(s['certified'])} skills + {len(s['gen'])} gen"
                for f, s in sorted(pool.items()))
            print(f"- final cap-probe pool per family: {pool_desc}")
        if name in ANCHOR_ROWS:
            print(f"- anchor_pen: mean {a['anchor_pen_mean']:.4f}, max {a['anchor_pen_max']:.4f}")
        fb = a["first_update_leaving_below_frozen"]
        if fb:
            fb_gen = "-" if fb["gen_rate"] is None else f"{fb['gen_rate']:.3f}"
            print(f"- first update leaving arith below frozen: v{fb['version']} "
                  f"({fb['family_phase']} phase, {fb['task_id']}): "
                  f"{fb['gold_prev_cand']:.3f} -> {fb['gold_cand']:.3f} "
                  f"(vs frozen {fb['gold_base']:.3f}), gen_rate={fb_gen}")
        else:
            print("- arith NEVER left below frozen")
        w = a["worst_single_update"]
        if w:
            w_gen = "-" if w["gen_rate"] is None else f"{w['gen_rate']:.3f}"
            print(f"- worst single update: v{w['version']} ({w['family_phase']} phase, "
                  f"{w['task_id']}) delta {w['gold_delta_vs_prev']:+.3f}, "
                  f"gen_rate={w_gen}")
        print(f"- final arith gold after last accepted update: {a['final_gold_cand']:.3f}")

        cols = ("| v | phase | task | gen wit | cap verdict | gen rate | cert rate "
                "| anchor pen | arith gold (prev -> cand) | d-prev |")
        print("\n" + cols)
        print("|--:|-------|------|---------|-------------|---------:|----------:|"
              "-----------:|---------------------------|-------:|")
        for r in a["timeline"]:
            verdict = ("pass" if not r["broke_cap"] else "VETO:" + ",".join(
                cap_source(x) for x in r["broke_cap"])) if r["checked_cap"] > 0 else "-"
            wit = ",".join(cap_source(x) for x in r["checked_gen"]) or "-"
            grate = "-" if r["gen_rate"] is None else f"{r['gen_rate']:.3f}"
            crate = "-" if r["cert_rate"] is None else f"{r['cert_rate']:.3f}"
            pen = f"{r['anchor_pen']:.3f}" if r["anchor_lambda"] > 0 else "-"
            print(f"| {r['version']} | {r['family_phase']} | {r['task_id']} | {wit} "
                  f"| {verdict} | {grate} | {crate} | {pen} "
                  f"| {r['gold_prev_cand']:.3f} -> {r['gold_cand']:.3f} "
                  f"| {r['gold_delta_vs_prev']:+.3f} |")
        print()

    # cross-row summary: the one-number view for the G1 mechanism reasoning
    print("### Cross-row summary (arith gold holdout, measurement only)\n")
    print("| learner | θ | erosion rate | M1 gen-sens | gen vetoes | final arith | cap vetoes |")
    print("|---------|--:|-------------:|------------:|-----------:|------------:|-----------:|")
    for name in LEARNERS:
        a = out[name]
        th = f"{THETA_ROWS[name]:.2f}" if name in THETA_ROWS else "0"
        s = a["m1_gen_witness_sensitivity"]
        m1 = f"{a['m1_gen_witness_seen']}/{a['probe_checked_erosions']}" + \
             (f" ({s:.0%})" if s is not None else "") \
             if a["probe_checked_erosions"] else "-"
        fin = f"{a['final_gold_cand']:.3f}" if a["final_gold_cand"] is not None else "-"
        print(f"| {name} | {th} | {a['gold_erosion_rate']:.1%} | {m1} "
              f"| {a['gate_stats']['cap_vetoes_by_gen']} | {fin} "
              f"| {a['gate_stats']['cap_vetoes']} |")
    print()

    dst = os.path.join(args.run, "telemetry_v8.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[telemetry] wrote {dst}")


if __name__ == "__main__":
    main()
