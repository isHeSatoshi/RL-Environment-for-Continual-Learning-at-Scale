"""SCCL v2 analyzer — leaderboard + per-family retention + gate/veto breakdown.

Reads metrics.json (and optional trajectories) from one or more run dirs and
prints a comparison. No gold is consumed here; this is post-hoc measurement.

Usage:
  python scripts/analyze_v2.py runs/sccl_v2 --ref runs/sccl_main
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _load_run(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return json.load(f)


def _fmt_row(name: str, d: Dict[str, Any]) -> str:
    r = d["report"]
    sc = d.get("sccl", {})
    return (
        f"| {name:<14} | {r['acc']:.3f} | {r['bwt']:+.3f} | {r['fwt']:+.3f} "
        f"| {r['forgetting']:.3f} | {d['frontier_score']:+.3f} "
        f"| {d['updates']:>3} | {d['rollbacks']:>2} "
        f"| {sc.get('cert_rate', 0.0):.2f} | {sc.get('probes_committed', 0):>2} |"
    )


def print_leaderboard(m: Dict[str, Any], ref: Dict[str, Any] | None, tag: str) -> None:
    print(f"\n=== Leaderboard: {tag} ===")
    hdr = ("| Learner        | ACC   | BWT    | FWT    | Forget | Frontier | Upd | RB "
           "| Cert | Pr |\n"
           "|----------------|-------|--------|--------|--------|----------|-----|----|------|----|")
    print(hdr)
    for name, d in m["learners"].items():
        print(_fmt_row(name, d))
    if ref:
        print(f"--- reference: {tag} (from ref run) ---")
        for name, d in ref["learners"].items():
            if name in m["learners"]:
                continue
            print(_fmt_row(name + "*", d))


def print_family_matrix(m: Dict[str, Any], fam_names: List[str], tag: str) -> None:
    print(f"\n=== Per-family retention (final_heldout / zero_shot): {tag} ===")
    head = "| Learner        | " + " | ".join(f"{fn:<18}" for fn in fam_names) + " |"
    sep = "|----------------|" + "|".join("--------------------" for _ in fam_names) + "|"
    print(head)
    print(sep)
    for name, d in m["learners"].items():
        rp = d.get("R_pairs", {})
        zs = rp.get("zero_shot", [])
        fh = rp.get("final", [])
        ft = rp.get("final_trained", [])
        cells = []
        for j, fn in enumerate(fam_names):
            z = zs[j] if j < len(zs) else 0.0
            h = fh[j] if j < len(fh) else 0.0
            t = ft[j] if j < len(ft) else 0.0
            cells.append(f"{h:.2f}/{z:.2f} (tr {t:.2f})")
        print(f"| {name:<14} | " + " | ".join(f"{c:<18}" for c in cells) + " |")


def print_sccl_stats(m: Dict[str, Any], tag: str) -> None:
    print(f"\n=== SCCL self-certification stats: {tag} ===")
    for name, d in m["learners"].items():
        sc = d.get("sccl", {})
        if not sc:
            continue
        print(
            f"  {name:<14} steps={sc.get('steps',0):>3} cert={sc.get('certified',0):>3} "
            f"rate={sc.get('cert_rate',0.0):.2f} conf={sc.get('mean_conf',0.0):.2f} "
            f"rrv_upd={sc.get('rrv_updates',0):>3} rrv_veto={sc.get('rrv_vetoes',0):>2} "
            f"probes_made={sc.get('probes_made',0):>2} probes_commit={sc.get('probes_committed',0):>2} "
            f"gold_agree={sc.get('gold_agreement',0.0):.2f}"
        )


def print_failure_detail(m: Dict[str, Any], fam_names: List[str], tag: str) -> None:
    """Per-task final-holdout failures (needs eval_detail; runs >= 2026-08-26)."""
    any_detail = any("eval_detail" in d for d in m["learners"].values())
    if not any_detail:
        return
    print(f"\n=== Failed final-holdout tasks: {tag} ===")
    for name, d in m["learners"].items():
        det = (d.get("eval_detail") or {}).get("final_heldout") or {}
        if not det:
            continue
        parts = []
        for fn in fam_names:
            recs = det.get(fn) or []
            failed = [r["task_id"] for r in recs if r.get("score", 0.0) < 0.999]
            if failed:
                parts.append(f"{fn}: {','.join(failed)}")
        print(f"  {name:<14} " + ("; ".join(parts) if parts else "all holdouts pass"))


def _gate_breakdown(traj_path: str) -> Dict[str, int]:
    counts = {"gated": 0, "accepted": 0, "vetoed": 0,
              "veto_skill": 0, "veto_probe": 0, "veto_math": 0,
              "probe_checked": 0, "math_checked": 0, "promoted": 0,
              "gold_ok": 0, "gold_total": 0}
    if not os.path.exists(traj_path):
        return counts
    with open(traj_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            gate = (rec.get("update_info") or {}).get("gate") or {}
            if gate.get("method") != "sccl_rrv":
                continue
            counts["gated"] += 1
            if gate.get("accepted"):
                counts["accepted"] += 1
            else:
                counts["vetoed"] += 1
                if gate.get("broke"):
                    counts["veto_skill"] += 1
                if gate.get("broke_probes"):
                    counts["veto_probe"] += 1
                if gate.get("broke_math"):
                    counts["veto_math"] += 1
            if gate.get("checked_probes"):
                counts["probe_checked"] += 1
            if gate.get("checked_math"):
                counts["math_checked"] += 1
            counts["promoted"] += int(gate.get("probes_promoted", 0))
            gt = gate.get("gold_telemetry")
            if isinstance(gt, dict):
                counts["gold_total"] += 1
                counts["gold_ok"] += int(bool(gt.get("gold_gate_ok")))
    return counts


def print_gate_breakdown(m: Dict[str, Any], run_dir: str, tag: str) -> None:
    print(f"\n=== RRV gate breakdown (from trajectories): {tag} ===")
    for name in m["learners"]:
        c = _gate_breakdown(os.path.join(run_dir, f"trajectories_{name}.jsonl"))
        if not c["gated"]:
            continue
        print(
            f"  {name:<14} gated={c['gated']:>3} accepted={c['accepted']:>3} "
            f"vetoed={c['vetoed']:>2} [skill={c['veto_skill']} probe={c['veto_probe']} "
            f"math={c['veto_math']}] probe_chk={c['probe_checked']:>3} math_chk={c['math_checked']:>3} "
            f"promoted={c['promoted']:>2} "
            f"gold_ok={c['gold_ok']}/{c['gold_total']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default=None, help="optional reference run dir")
    args = ap.parse_args()

    m = _load_run(args.run_dir)
    ref = _load_run(args.ref) if args.ref else None
    fam_names = list(m.get("config", {}).get("families", []) or [])
    if not fam_names:
        # derive ordered family names from any learner's trajectory stream
        for name in m["learners"]:
            tp = os.path.join(args.run_dir, f"trajectories_{name}.jsonl")
            if not os.path.exists(tp):
                continue
            seen: List[str] = []
            with open(tp) as f:
                for line in f:
                    try:
                        fn = json.loads(line).get("family")
                    except Exception:
                        continue
                    if fn and fn not in seen:
                        seen.append(fn)
            if seen:
                fam_names = seen
                break
    if not fam_names:
        for d in m["learners"].values():
            zs = d.get("R_pairs", {}).get("zero_shot", [])
            fam_names = [f"f{i}" for i in range(len(zs))]
            break

    tag = os.path.basename(os.path.normpath(args.run_dir))
    print_leaderboard(m, ref, tag)
    print_family_matrix(m, fam_names, tag)
    print_failure_detail(m, fam_names, tag)
    print_sccl_stats(m, tag)
    print_gate_breakdown(m, args.run_dir, tag)
    if ref and args.ref:
        rtag = os.path.basename(os.path.normpath(args.ref))
        print_family_matrix(ref, fam_names, rtag + " (ref)")


if __name__ == "__main__":
    main()
