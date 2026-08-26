"""Post-hoc audit of the v3 admission-time neighborhood filter.

Gold is MEASUREMENT ONLY here: for SCCL learners the action fed to env.step
IS the certified winner, so the trajectory's `verifier` block (gold unit-test
results, logged as reward telemetry) tells us whether each winner was
gold-correct. We correlate the gold-free admission decision (sccl_nbhd) with
that gold outcome:

  * over-filter rate   : rejected by the filter BUT gold-correct
                         (evidence the filter is too aggressive / paraphrase
                         drifted)
  * correct-skepticism : rejected AND gold-wrong (the filter caught a bad
                         target the point-cert had accepted)
  * admitted-gold-wrong: admitted but gold-wrong (residual error any
                         gold-free method pays)

Usage:
  python scripts/analyze_nbhd.py --run runs/sccl_v3_s42 [--learner sccl_n]
  python scripts/analyze_nbhd.py --run runs/sccl_v3_s42 --baseline sccl
      (--baseline adds the unfiltered v1-reference row's gold stats for
       comparison; it never touches any decision path)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _rows(run_dir: str, learner: str):
    p = os.path.join(run_dir, f"trajectories_{learner}.jsonl")
    if not os.path.exists(p):
        print(f"missing {p}")
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _gold_ok(row: dict) -> float | None:
    v = row.get("verifier") or {}
    pr = v.get("pass_rate")
    return float(pr) if pr is not None else None


def _trained(row: dict) -> bool:
    ui = row.get("update_info") or {}
    return bool(ui.get("executed")) and bool(ui.get("accepted", True))


def report(run_dir: str, learner: str) -> None:
    rows = _rows(run_dir, learner)
    if not rows:
        return
    checked = [r for r in rows if isinstance(r.get("sccl_nbhd"), dict)]
    print(f"=== nbhd admission audit: {learner} ({len(checked)}/{len(rows)} steps checked) ===")
    if not checked:
        print("  (no neighborhood checks recorded)")
        return

    buckets = {"admitted": [], "rejected": []}
    reasons: dict[str, list[float]] = {}
    for r in checked:
        nb = r["sccl_nbhd"]
        gold = _gold_ok(r)
        key = "admitted" if nb.get("robust", True) else "rejected"
        if gold is not None:
            buckets[key].append(gold)
        reason = nb.get("reason", "") or "ok"
        if gold is not None:
            reasons.setdefault(reason, []).append(gold)

    for key, gs in buckets.items():
        if gs:
            print(f"  {key:<9} n={len(gs):>3}  mean gold pass_rate={sum(gs)/len(gs):.3f}  "
                  f"gold-perfect={sum(1 for g in gs if g >= 1.0)}/{len(gs)}")
    rej = buckets["rejected"]
    adm = buckets["admitted"]
    if rej:
        print(f"  over-filter rate (rejected & gold==1.0): "
              f"{sum(1 for g in rej if g >= 1.0)}/{len(rej)}")
        print(f"  correct-skepticism  (rejected & gold<1.0): "
              f"{sum(1 for g in rej if g < 1.0)}/{len(rej)}")
    if adm:
        print(f"  admitted-gold-wrong (admitted & gold<1.0): "
              f"{sum(1 for g in adm if g < 1.0)}/{len(adm)}")
    print("  by reason (mean gold pass_rate, n):")
    for reason, gs in sorted(reasons.items()):
        print(f"    {reason:<32} {sum(gs)/len(gs):.3f}  n={len(gs)}")


def baseline(run_dir: str, learner: str) -> None:
    rows = _rows(run_dir, learner)
    if not rows:
        return
    gs = [g for g in (_gold_ok(r) for r in rows) if g is not None]
    cert = [r for r in rows if (r.get("sccl") or {}).get("found")]
    gsc = [g for g in (_gold_ok(r) for r in cert) if g is not None]
    print(f"=== baseline (no admission filter): {learner} ===")
    if gs:
        print(f"  all steps      n={len(gs):>3}  mean gold pass_rate={sum(gs)/len(gs):.3f}")
    if gsc:
        print(f"  certified only n={len(gsc):>3}  mean gold pass_rate={sum(gsc)/len(gsc):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--learner", default="sccl_n")
    ap.add_argument("--baseline", default="", help="learner row without the filter (e.g. sccl)")
    args = ap.parse_args()
    report(args.run, args.learner)
    if args.baseline:
        baseline(args.run, args.baseline)


if __name__ == "__main__":
    sys.exit(main())
