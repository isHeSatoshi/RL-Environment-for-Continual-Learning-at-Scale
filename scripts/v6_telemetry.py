"""Post-hoc gold telemetry for the v6 ensemble/anchor ladder (MEASUREMENT ONLY).

Extends scripts/v5b_telemetry.py with the two v6 interventions:

  1. ENSEMBLE engagement: per-gate checked_cap / broke_cap distributions from
     the trajectory gate records, plus the final cap-probe pool composition per
     family (distinct source skills) from each learner's vault skills.json.
     Answers: did the K=3 pool actually witness more skill axes, and did it
     veto updates that the K=1 pool passed (the v5b mechanism failure)?
  2. ANCHOR engagement: anchor_pen timeline (update_info top level, accepted
     updates only) and its correlation with gold erosion — did the quadratic
     pull toward the LoRA init reduce the rate of accepted updates that erode
     the arith gold holdout?

Gold scores were logged by the gate at run time but NEVER entered any
accept/reject decision; this script only tabulates them after the fact, same
protocol as v5b.

Usage: python scripts/v6_telemetry.py [--run runs/sccl_v6]
Writes <run>/telemetry_v6.json and prints a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


LEARNERS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat",
            "sccl_capens", "sccl_cap_anchor", "sccl_capens_anchor"]
ANCHOR_ROWS = {"sccl_cap_anchor", "sccl_capens_anchor"}
ENSEMBLE_ROWS = {"sccl_capens", "sccl_capens_anchor"}
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
    """family -> distinct source skills among stored cap_probe entries."""
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
        out.setdefault(fam, set()).add(cap_source(r["task_id"]))
    return {f: sorted(s) for f, s in out.items()}


def gate_stats(run_dir: str, name: str) -> dict:
    """Gate-level cap-probe engagement from trajectory records (all updates)."""
    checked_hist: Counter = Counter()
    vetoed: list = []
    n_gates = 0
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        g = ui.get("gate") or {}
        if "checked_cap" not in g:
            continue
        n_gates += 1
        checked_hist[int(g.get("checked_cap", 0))] += 1
        if g.get("broke_cap"):
            vetoed.append({
                "family": rec.get("family", "?"),
                "task_id": rec.get("task_id", "?"),
                "checked_cap": int(g.get("checked_cap", 0)),
                "broke_cap": list(g.get("broke_cap", [])),
                "cap_guard_armed": bool(g.get("cap_guard_armed", False)),
            })
    return {
        "gates_with_cap_records": n_gates,
        "checked_cap_histogram": dict(sorted(checked_hist.items())),
        "max_checked_cap": max(checked_hist) if checked_hist else 0,
        "cap_vetoes": len(vetoed),
        "cap_veto_events": vetoed,
    }


def analyze(run_dir: str, name: str) -> dict:
    # adapter_version -> (family, task_id) from trajectory update records
    fam_of: dict = {}
    anchor_pen_of: dict = {}
    anchor_lam_of: dict = {}
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        v = ui.get("adapter_version")
        if v is not None and ui.get("accepted"):
            fam_of[int(v)] = (rec.get("family", "?"), rec.get("task_id", "?"))
            anchor_pen_of[int(v)] = float(ui.get("anchor_pen", 0.0) or 0.0)
            anchor_lam_of[int(v)] = float(ui.get("anchor_lambda", 0.0) or 0.0)

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
        rows.append({
            "version": int(m["version"]),
            "family_phase": fam,
            "task_id": task,
            "checked_cap": int(g.get("checked_cap", 0)),
            "broke_cap": list(g.get("broke_cap", [])),
            "cap_guard_armed": bool(g.get("cap_guard_armed", False)),
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
    ap.add_argument("--run", default="runs/sccl_v6")
    args = ap.parse_args()

    out: dict = {}
    print(f"## Post-hoc gold telemetry — v6 ensemble/anchor ladder ({args.run})\n")
    print("Gold scores were logged by the gate at run time but NEVER used in any\n"
          "accept/reject decision. `base` = frozen arith gold holdout (cached);\n"
          "`cand` = candidate adapter's score, i.e. the arith level each ACCEPTED\n"
          "update left behind.\n")

    for name in LEARNERS:
        a = analyze(args.run, name)
        a["gate_stats"] = gate_stats(args.run, name)
        a["final_cap_pool"] = vault_pool(args.run, name)
        out[name] = a

        if not a["timeline"]:
            print(f"### {name}: no gold telemetry rows\n")
            continue
        tag = []
        if name in ENSEMBLE_ROWS:
            tag.append("ENSEMBLE K=3")
        if name in ANCHOR_ROWS:
            tag.append("ANCHOR λ=0.1")
        hdr = f"### {name}" + (f"  ({', '.join(tag)})" if tag else "")
        print(hdr)
        print(f"- accepted updates w/ telemetry: {a['accepted_updates_with_gold_telemetry']}")
        print(f"- gold-erosion updates (cand < prev - {EPS}): "
              f"**{a['gold_erosion_updates']}** ({a['gold_erosion_rate']:.1%})")
        if a["insensitivity_rate"] is not None:
            print(f"- probe-checked erosions: {a['probe_checked_erosions']}; probe "
                  f"**PASSED** despite erosion: **{a['probe_PASSED_despite_erosion']}** "
                  f"(insensitivity {a['insensitivity_rate']:.1%})")
        else:
            print("- no probe-checked erosions (probes off or none checked)")
        gs = a["gate_stats"]
        print(f"- cap gates: {gs['gates_with_cap_records']}; checked_cap histogram "
              f"{gs['checked_cap_histogram']}; max checked {gs['max_checked_cap']}; "
              f"cap-vetoed updates {gs['cap_vetoes']}")
        pool = a["final_cap_pool"]
        if pool:
            pool_desc = "; ".join(f"{f}:{len(s)} skills" for f, s in sorted(pool.items()))
            print(f"- final cap-probe pool per family: {pool_desc}")
        if name in ANCHOR_ROWS:
            print(f"- anchor_pen: mean {a['anchor_pen_mean']:.4f}, max {a['anchor_pen_max']:.4f}")
        fb = a["first_update_leaving_below_frozen"]
        if fb:
            print(f"- first update leaving arith below frozen: v{fb['version']} "
                  f"({fb['family_phase']} phase, {fb['task_id']}): "
                  f"{fb['gold_prev_cand']:.3f} -> {fb['gold_cand']:.3f} "
                  f"(vs frozen {fb['gold_base']:.3f}), "
                  f"probe passed={fb['checked_cap'] > 0 and not fb['broke_cap']}")
        else:
            print("- arith NEVER left below frozen")
        w = a["worst_single_update"]
        if w:
            print(f"- worst single update: v{w['version']} ({w['family_phase']} phase, "
                  f"{w['task_id']}) delta {w['gold_delta_vs_prev']:+.3f}, "
                  f"probe passed={w['checked_cap'] > 0 and not w['broke_cap']}")
        print(f"- final arith gold after last accepted update: {a['final_gold_cand']:.3f}")
        print("\n| v | phase | task | cap chk | cap verdict | anchor pen | arith gold (prev -> cand) | d-prev | vs frozen |")
        print("|--:|-------|------|--------:|-------------|-----------:|---------------------------|-------:|----------:|")
        for r in a["timeline"]:
            verdict = ("pass" if not r["broke_cap"] else "VETO:" + ",".join(
                cap_source(x) for x in r["broke_cap"])) if r["checked_cap"] > 0 else "-"
            pen = f"{r['anchor_pen']:.3f}" if r["anchor_lambda"] > 0 else "-"
            print(f"| {r['version']} | {r['family_phase']} | {r['task_id']} | {r['checked_cap']} "
                  f"| {verdict} | {pen} | {r['gold_prev_cand']:.3f} -> {r['gold_cand']:.3f} "
                  f"| {r['gold_delta_vs_prev']:+.3f} | {r['gold_delta_vs_frozen']:+.3f} |")
        print()

    # cross-row summary: the one-number view for H1/H2/H3 mechanism reasoning
    print("### Cross-row summary (arith gold holdout, measurement only)\n")
    print("| learner | erosion rate | insensitivity | final arith gold | below-frozen updates | cap vetoes |")
    print("|---------|-------------:|--------------:|-----------------:|---------------------:|-----------:|")
    for name in LEARNERS:
        a = out[name]
        ins = f"{a['insensitivity_rate']:.1%}" if a["insensitivity_rate"] is not None else "-"
        fin = f"{a['final_gold_cand']:.3f}" if a["final_gold_cand"] is not None else "-"
        print(f"| {name} | {a['gold_erosion_rate']:.1%} | {ins} | {fin} "
              f"| {a['updates_leaving_arith_below_frozen']} | {a['gate_stats']['cap_vetoes']} |")
    print()

    dst = os.path.join(args.run, "telemetry_v6.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[telemetry] wrote {dst}")


if __name__ == "__main__":
    main()
