"""Post-hoc gold telemetry for the v7 ladder (Branch E, E1 pass-rate margin veto).

MEASUREMENT ONLY. Gold scores were logged by the gate at run time but NEVER
entered any accept/reject decision; this script tabulates them after the fact,
same protocol as v5b/v6. It extends scripts/v6_telemetry.py with the E1
evidence layer:

  1. E1 RATE SENSITIVITY. The v6 verdict found 100% probe insensitivity: every
     probe-checked gold erosion PASSED the binary any-of-n retain rule, because
     the rule collapses n draws to a boolean and cannot see rate degradation.
     E1 logs per-probe draw outcomes (gate["cap_rates"] = {task_id: {passes, n}})
     on theta>0 rows. The central telemetry question: at the gates that ACCEPTED
     an update which later eroded the arith gold holdout, what was the pooled
     pass rate? If eroding gates show rate < 1.0, the signal existed and a
     stricter theta could have caught it; if they all show rate == 1.0, the
     probe pool itself is blind (n draws too few / wrong axes) and no threshold
     fixes it — that is the E1 mechanism failure mode and routes to E2.
  2. DOSE RESPONSE. Per-row cap-veto counts and pooled-rate distributions for
     the theta ladder (baseline theta=0 any-pass, majority theta=2/3, strict
     theta=1.0) — the observational counterpart of pre-registered H1b.
  3. Everything from v6: erosion timelines, insensitivity, anchor engagement,
     cap-probe pool composition.

Usage: python scripts/v7_telemetry.py [--run runs/sccl_v7]
Writes <run>/telemetry_v7.json and prints a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


LEARNERS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat",
            "sccl_strict", "sccl_majority",
            "sccl_ens_strict", "sccl_ens_strict_anchor"]
THETA_ROWS = {"sccl_strict": 1.0, "sccl_majority": 2.0 / 3.0,
              "sccl_ens_strict": 1.0, "sccl_ens_strict_anchor": 1.0}
ANCHOR_ROWS = {"sccl_ens_strict_anchor"}
ENSEMBLE_ROWS = {"sccl_ens_strict", "sccl_ens_strict_anchor"}
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


def pooled_rate(cap_rates: dict) -> float | None:
    """sum(passes)/sum(n) over the per-probe entries of one gate record."""
    if not cap_rates:
        return None
    passes = sum(int(v.get("passes", 0)) for v in cap_rates.values())
    n = sum(int(v.get("n", 0)) for v in cap_rates.values())
    return passes / n if n > 0 else None


def gate_stats(run_dir: str, name: str) -> dict:
    """Gate-level engagement from trajectory records (all updates, kept+vetoes)."""
    checked_hist: Counter = Counter()
    rate_hist: Counter = Counter()
    vetoed: list = []
    rate_of_kept: list = []
    n_gates = 0
    n_e1 = 0
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        g = ui.get("gate") or {}
        if "checked_cap" not in g:
            continue
        n_gates += 1
        checked_hist[int(g.get("checked_cap", 0))] += 1
        rates = g.get("cap_rates") or {}
        pr = pooled_rate(rates)
        if rates:
            n_e1 += 1
            # bin pooled rates to thirds for the distribution view
            rate_hist[round(pr * 3) / 3 if pr is not None else -1] += 1
            if g.get("broke_cap"):
                vetoed.append({
                    "family": rec.get("family", "?"),
                    "task_id": rec.get("task_id", "?"),
                    "pooled_rate": pr,
                    "cap_rates": rates,
                    "broke_cap": list(g.get("broke_cap", [])),
                    "cap_guard_armed": bool(g.get("cap_guard_armed", False)),
                })
            elif ui.get("accepted") and pr is not None:
                rate_of_kept.append(pr)
    return {
        "gates_with_cap_records": n_gates,
        "e1_gates_with_rates": n_e1,
        "checked_cap_histogram": dict(sorted(checked_hist.items())),
        "max_checked_cap": max(checked_hist) if checked_hist else 0,
        "pooled_rate_histogram_thirds": dict(sorted(
            (k, v) for k, v in rate_hist.items() if k >= 0)),
        "kept_accepted_rates_min": min(rate_of_kept, default=None),
        "kept_accepted_rates_mean": (sum(rate_of_kept) / len(rate_of_kept)
                                     if rate_of_kept else None),
        "cap_vetoes": len(vetoed),
        "cap_veto_events": vetoed,
    }


def analyze(run_dir: str, name: str) -> dict:
    # adapter_version -> (family, task_id, pooled_rate) from accepted updates
    fam_of: dict = {}
    anchor_pen_of: dict = {}
    anchor_lam_of: dict = {}
    gate_rate_of: dict = {}
    gate_detail_of: dict = {}
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        v = ui.get("adapter_version")
        if v is not None and ui.get("accepted"):
            fam_of[int(v)] = (rec.get("family", "?"), rec.get("task_id", "?"))
            anchor_pen_of[int(v)] = float(ui.get("anchor_pen", 0.0) or 0.0)
            anchor_lam_of[int(v)] = float(ui.get("anchor_lambda", 0.0) or 0.0)
            g = ui.get("gate") or {}
            rates = g.get("cap_rates") or {}
            if rates:
                gate_rate_of[int(v)] = pooled_rate(rates)
                gate_detail_of[int(v)] = rates

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
            "e1_pooled_rate": gate_rate_of.get(int(m["version"])),
            "e1_rates": gate_detail_of.get(int(m["version"])),
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
    # E1 signal view: among probe-checked EROSIONS that were ACCEPTED, how many
    # had pooled rate < 1.0 (a visible warning) vs rate == 1.0 (probe blind)?
    er_with_rate = [r for r in probe_passed if r["e1_pooled_rate"] is not None]
    er_rate_below_1 = [r for r in er_with_rate if r["e1_pooled_rate"] < 1.0 - 1e-9]
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
        "e1_erosion_accepted_with_rate": len(er_with_rate),
        "e1_erosion_rate_below_1": len(er_rate_below_1),
        "e1_erosion_rate_blind_1.0": len(er_with_rate) - len(er_rate_below_1),
        "e1_erosion_rates": sorted(
            round(r["e1_pooled_rate"], 3) for r in er_with_rate),
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
    ap.add_argument("--run", default="runs/sccl_v7")
    args = ap.parse_args()

    out: dict = {}
    print(f"## Post-hoc gold telemetry — v7 Branch E ladder ({args.run})\n")
    print("Gold scores were logged by the gate at run time but NEVER used in any\n"
          "accept/reject decision. `base` = frozen arith gold holdout (cached);\n"
          "`cand` = candidate adapter's score, i.e. the arith level each ACCEPTED\n"
          "update left behind. `e1_pooled_rate` = sum(passes)/sum(n) over the\n"
          "gate's cap_rates entries (theta>0 rows only).\n")

    for name in LEARNERS:
        a = analyze(args.run, name)
        a["gate_stats"] = gate_stats(args.run, name)
        a["final_cap_pool"] = vault_pool(args.run, name)
        out[name] = a

        if not a["timeline"]:
            print(f"### {name}: no gold telemetry rows\n")
            continue
        tag = []
        if name in THETA_ROWS:
            tag.append(f"E1 θ={THETA_ROWS[name]:.3f}")
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
                  f"(binary insensitivity {a['insensitivity_rate']:.1%})")
            if a["e1_erosion_accepted_with_rate"]:
                print(f"- E1 signal on accepted erosions: "
                      f"{a['e1_erosion_rate_below_1']}/{a['e1_erosion_accepted_with_rate']} "
                      f"had pooled rate < 1.0 (visible warning) | rates {a['e1_erosion_rates']} "
                      f"| {a['e1_erosion_rate_blind_1.0']} at rate 1.0 = probe-blind")
        else:
            print("- no probe-checked erosions (probes off or none checked)")
        gs = a["gate_stats"]
        print(f"- cap gates: {gs['gates_with_cap_records']} (E1 with rates: "
              f"{gs['e1_gates_with_rates']}); checked_cap histogram "
              f"{gs['checked_cap_histogram']}; cap-vetoed updates {gs['cap_vetoes']}")
        if gs["pooled_rate_histogram_thirds"]:
            print(f"- pooled-rate histogram (thirds): {gs['pooled_rate_histogram_thirds']}; "
                  f"kept+accepted rate min={gs['kept_accepted_rates_min']}, "
                  f"mean={gs['kept_accepted_rates_mean']}")
        pool = a["final_cap_pool"]
        if pool:
            pool_desc = "; ".join(f"{f}:{len(s)} skills" for f, s in sorted(pool.items()))
            print(f"- final cap-probe pool per family: {pool_desc}")
        if name in ANCHOR_ROWS:
            print(f"- anchor_pen: mean {a['anchor_pen_mean']:.4f}, max {a['anchor_pen_max']:.4f}")
        fb = a["first_update_leaving_below_frozen"]
        if fb:
            r = fb["e1_pooled_rate"]
            print(f"- first update leaving arith below frozen: v{fb['version']} "
                  f"({fb['family_phase']} phase, {fb['task_id']}): "
                  f"{fb['gold_prev_cand']:.3f} -> {fb['gold_cand']:.3f} "
                  f"(vs frozen {fb['gold_base']:.3f}), "
                  f"e1_rate={'-' if r is None else f'{r:.3f}'}")
        else:
            print("- arith NEVER left below frozen")
        w = a["worst_single_update"]
        if w:
            r = w["e1_pooled_rate"]
            print(f"- worst single update: v{w['version']} ({w['family_phase']} phase, "
                  f"{w['task_id']}) delta {w['gold_delta_vs_prev']:+.3f}, "
                  f"e1_rate={'-' if r is None else f'{r:.3f}'}")
        print(f"- final arith gold after last accepted update: {a['final_gold_cand']:.3f}")

        cols = "| v | phase | task | cap chk | cap verdict | e1 rate | anchor pen | arith gold (prev -> cand) | d-prev |"
        print("\n" + cols)
        print("|--:|-------|------|--------:|-------------|--------:|-----------:|---------------------------|-------:|")
        for r in a["timeline"]:
            verdict = ("pass" if not r["broke_cap"] else "VETO:" + ",".join(
                cap_source(x) for x in r["broke_cap"])) if r["checked_cap"] > 0 else "-"
            rate = "-" if r["e1_pooled_rate"] is None else f"{r['e1_pooled_rate']:.3f}"
            pen = f"{r['anchor_pen']:.3f}" if r["anchor_lambda"] > 0 else "-"
            print(f"| {r['version']} | {r['family_phase']} | {r['task_id']} | {r['checked_cap']} "
                  f"| {verdict} | {rate} | {pen} | {r['gold_prev_cand']:.3f} -> {r['gold_cand']:.3f} "
                  f"| {r['gold_delta_vs_prev']:+.3f} |")
        print()

    # cross-row summary: the one-number view for the E1 mechanism reasoning
    print("### Cross-row summary (arith gold holdout, measurement only)\n")
    print("| learner | θ | erosion rate | insens. | er. w/ rate<1.0 | final arith | cap vetoes |")
    print("|---------|--:|-------------:|--------:|----------------:|------------:|-----------:|")
    for name in LEARNERS:
        a = out[name]
        th = f"{THETA_ROWS[name]:.2f}" if name in THETA_ROWS else "0"
        ins = f"{a['insensitivity_rate']:.1%}" if a["insensitivity_rate"] is not None else "-"
        fin = f"{a['final_gold_cand']:.3f}" if a["final_gold_cand"] is not None else "-"
        sig = (f"{a['e1_erosion_rate_below_1']}/{a['e1_erosion_accepted_with_rate']}"
               if a["e1_erosion_accepted_with_rate"] else "-")
        print(f"| {name} | {th} | {a['gold_erosion_rate']:.1%} | {ins} | {sig} | {fin} "
              f"| {a['gate_stats']['cap_vetoes']} |")
    print()

    dst = os.path.join(args.run, "telemetry_v7.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[telemetry] wrote {dst}")


if __name__ == "__main__":
    main()
