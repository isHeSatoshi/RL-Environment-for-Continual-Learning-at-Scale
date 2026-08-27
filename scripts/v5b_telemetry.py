"""Post-hoc gold telemetry for the v5b capability-probe ladder (MEASUREMENT ONLY).

Reads the gate records already logged during runs/sccl_v5b: every accepted
update of every SCCL learner was scored on the arith gold holdout before
(base) and after (cand) the candidate update (env.py gold_telemetry block).
Those scores NEVER entered any accept/reject decision; this script only
tabulates them after the fact to answer:

  1. Which accepted updates eroded the arith holdout?
  2. For each erosion, did the capability-probe stratum PASS the update
     (checked_cap>0, broke_cap empty)?  -> quantifies probe insensitivity.
  3. Where in the stream (which family phase) does erosion happen?

Usage: python scripts/v5b_telemetry.py [--run runs/sccl_v5b]
Writes <run>/telemetry_arith_erosion.json and prints a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import os


LEARNERS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat"]
EPS = 0.05


def load_registry(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, "adapters_" + name, "registry.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f).get("metas", [])


def load_updates(run_dir: str, name: str) -> list:
    p = os.path.join(run_dir, f"trajectories_{name}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def analyze(run_dir: str, name: str) -> dict:
    # adapter_version -> (family, task_id) from trajectory update records
    fam_of: dict = {}
    for rec in load_updates(run_dir, name):
        ui = rec.get("update_info") or {}
        v = ui.get("adapter_version")
        if v is not None and ui.get("accepted"):
            fam_of[int(v)] = (rec.get("family", "?"), rec.get("task_id", "?"))

    rows = []
    # `base` is the cached FROZEN score (adapter off, measured once at the
    # first gate), so also track the per-update change vs the previous
    # accepted version to attribute erosion to a specific update.
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
            prev_cand = base  # before the first update, the model IS the base
        rows.append({
            "version": int(m["version"]),
            "family_phase": fam,
            "task_id": task,
            "checked_cap": int(g.get("checked_cap", 0)),
            "broke_cap": list(g.get("broke_cap", [])),
            "cap_guard_armed": bool(g.get("cap_guard_armed", False)),
            "checked": int(g.get("checked", 0)),
            "broke": list(g.get("broke", [])),
            "gold_base": base,
            "gold_prev_cand": prev_cand,
            "gold_cand": cand,
            "gold_delta_vs_frozen": cand - base,
            "gold_delta_vs_prev": cand - prev_cand,
            "gold_gate_ok": bool(gt["gold_gate_ok"]),
        })
        prev_cand = cand

    erosion = [r for r in rows if r["gold_delta_vs_prev"] < -EPS]
    probe_checked_erosion = [r for r in erosion if r["checked_cap"] > 0]
    probe_passed_erosion = [r for r in probe_checked_erosion if not r["broke_cap"]]
    below_frozen = [r for r in rows if r["gold_delta_vs_frozen"] < -EPS]
    first_below = next((r for r in rows if r["gold_delta_vs_frozen"] < -EPS), None)
    worst = min(rows, key=lambda r: r["gold_delta_vs_prev"]) if rows else None
    return {
        "learner": name,
        "accepted_updates_with_gold_telemetry": len(rows),
        "gold_erosion_updates": len(erosion),
        "gold_erosion_rate": round(len(erosion) / max(1, len(rows)), 3),
        "probe_checked_erosions": len(probe_checked_erosion),
        "probe_PASSED_despite_erosion": len(probe_passed_erosion),
        "insensitivity_rate": (round(len(probe_passed_erosion) / len(probe_checked_erosion), 3)
                               if probe_checked_erosion else None),
        "updates_leaving_arith_below_frozen": len(below_frozen),
        "first_update_leaving_below_frozen": first_below,
        "worst_single_update": worst,
        "final_gold_cand": rows[-1]["gold_cand"] if rows else None,
        "timeline": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/sccl_v5b")
    args = ap.parse_args()

    out = {}
    for name in LEARNERS:
        out[name] = analyze(args.run, name)

    print(f"## Post-hoc gold telemetry — arith holdout erosion ({args.run})\n")
    print("Gold scores were logged by the gate at run time but NEVER used in any\n"
          "accept/reject decision. `base` = frozen arith gold holdout (cached at the\n"
          "first gate); `cand` = candidate adapter's score on the same holdout, i.e.\n"
          "the arith level each ACCEPTED update left behind.\n")
    for name, a in out.items():
        if not a["timeline"]:
            print(f"### {name}: no gold telemetry rows\n")
            continue
        print(f"### {name}")
        print(f"- accepted updates w/ telemetry: {a['accepted_updates_with_gold_telemetry']}")
        print(f"- gold-erosion updates (cand < base - {EPS}): "
              f"**{a['gold_erosion_updates']}** ({a['gold_erosion_rate']:.1%})")
        if a["insensitivity_rate"] is not None:
            print(f"- of those, cap-probe checked: {a['probe_checked_erosions']}; "
                  f"probe **PASSED** despite erosion: **{a['probe_PASSED_despite_erosion']}** "
                  f"(insensitivity {a['insensitivity_rate']:.1%})")
        else:
            print("- no probe-checked erosions (probes off or none checked)")
        fb = a["first_update_leaving_below_frozen"]
        if fb:
            print(f"- first update leaving arith below frozen: v{fb['version']} "
                  f"({fb['family_phase']} phase, {fb['task_id']}): "
                  f"{fb['gold_prev_cand']:.3f} -> {fb['gold_cand']:.3f} "
                  f"(vs frozen {fb['gold_base']:.3f}), "
                  f"probe passed={fb['checked_cap'] > 0 and not fb['broke_cap']}")
        print(f"- accepted updates leaving arith below frozen: "
              f"{a['updates_leaving_arith_below_frozen']}/{a['accepted_updates_with_gold_telemetry']}")
        w = a["worst_single_update"]
        print(f"- worst single update: v{w['version']} ({w['family_phase']} phase, {w['task_id']}) "
              f"delta {w['gold_delta_vs_prev']:+.3f}, probe passed={w['checked_cap'] > 0 and not w['broke_cap']}")
        print(f"- final arith gold after last accepted update: {a['final_gold_cand']:.3f}")
        print("\n| v | phase | task | cap chk | cap verdict | arith gold (prev -> cand) | d-prev | vs frozen | gold-gate |")
        print("|--:|-------|------|--------:|-------------|---------------------------|-------:|----------:|-----------|")
        for r in a["timeline"]:
            verdict = ("pass" if not r["broke_cap"] else "VETO:" + ",".join(r["broke_cap"])) \
                if r["checked_cap"] > 0 else "-"
            print(f"| {r['version']} | {r['family_phase']} | {r['task_id']} | {r['checked_cap']} "
                  f"| {verdict} | {r['gold_prev_cand']:.3f} -> {r['gold_cand']:.3f} "
                  f"| {r['gold_delta_vs_prev']:+.3f} | {r['gold_delta_vs_frozen']:+.3f} "
                  f"| {'ok' if r['gold_gate_ok'] else 'REJECT'} |")
        print()

    dst = os.path.join(args.run, "telemetry_arith_erosion.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[telemetry] wrote {dst}")


if __name__ == "__main__":
    main()
