"""Detached watcher + analyzer for the seeded SCCL v4 ladder (runs/sccl_v4).

Waits for the runner to finish (metrics.json with all 5 learners), then
evaluates the PRE-REGISTERED v4 decision rule (RESEARCH_NOTES_v4.md) and
writes runs/sccl_v4_verdict.md. Never touches gold in any decision here:
gold fields below are measurement-only cross-checks of determinism.

  * success -> analyze + write verdict + marker, exit 0
  * crash   -> runner gone, metrics missing -> exit 2
  * timeout -> wall-clock cap exceeded -> exit 3
Runs detached; the agent is re-woken on completion.
"""
import json
import os
import subprocess
import sys
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_tmp = "D:/gcl_tmp"
os.makedirs(_tmp, exist_ok=True)
for k in ("TMPDIR", "TEMP", "TMP"):
    os.environ[k] = _tmp

METRICS = "runs/sccl_v4/metrics.json"
VERDICT = "runs/sccl_v4_verdict.md"
MARKER = "runs/sccl_v4_watchdone.txt"
MAX_WAIT = 6 * 3600
INTERVAL = 120
LEARNERS = ["frozen", "sccl", "sccl_anchor_lo", "sccl_anchor_hi", "vsr_nogold"]

# v3 seed-42 reference rows (same learner indices -> same RNG seeds) for the
# determinism cross-check. Measurement-only.
V3_REF = {
    "sccl":       {"acc": 0.575, "forgetting": 0.075, "frontier": 0.500, "arith": 0.400},
    "vsr_nogold": {"acc": 0.637, "forgetting": 0.125, "frontier": 0.512, "arith": 0.400},
}


def runner_count() -> int:
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*sccl_v4.json*' } | "
           "Measure-Object | Select-Object -ExpandProperty Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=60)
        s = out.stdout.strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return -1


def load_metrics():
    try:
        m = json.load(open(METRICS))
        if "learners" in m and all(k in m["learners"] for k in LEARNERS):
            return m
    except Exception:
        pass
    return None


def fam_acc(lr, fam):
    try:
        ts = lr["eval_detail"]["final_heldout"][fam]
        return sum(t["score"] for t in ts) / len(ts)
    except Exception:
        return None


def audit_anchor(learner):
    """Read persisted gate audit trail: list of anchor_lambda per accepted update."""
    path = f"runs/sccl_v4/adapters_{learner}/registry.json"
    try:
        d = json.load(open(path))
        metas = d if isinstance(d, list) else d.get("metas", [])
        out = []
        for mm in metas:
            g = mm.get("gate") or {}
            if "anchor_lambda" in g:
                out.append(g["anchor_lambda"])
        return out
    except Exception:
        return None


def analyze(m):
    # ---------------- analysis (pre-registered rule) ----------------
    rows = {}
    for name in LEARNERS:
        lr = m["learners"][name]
        rep = lr["report"]
        rows[name] = {
            "acc": rep["acc"], "bwt": rep["bwt"], "forgetting": rep["forgetting"],
            "frontier": lr["frontier_score"], "updates": lr["updates"],
            "rollbacks": lr["rollbacks"],
            "arith": fam_acc(lr, "arith"), "math": fam_acc(lr, "math_word"),
            "string": fam_acc(lr, "string"), "drift": fam_acc(lr, "drift"),
        }

    sccl = rows["sccl"]
    lines = []
    lines.append("# SCCL v4 verdict (seed 42 ladder, runs/sccl_v4) — pre-registered rule")
    lines.append("")
    lines.append("| learner | ACC | BWT | forget | frontier | upd/rbk | arith-hold | math | string | drift |")
    lines.append("|---------|-----|-----|--------|----------|---------|-----------|------|--------|-------|")
    for name in LEARNERS:
        r = rows[name]
        lines.append("| %s | %.3f | %+.3f | %.3f | %+.3f | %d/%d | %s | %s | %s | %s |" % (
            name, r["acc"], r["bwt"], r["forgetting"], r["frontier"],
            r["updates"], r["rollbacks"],
            "%.3f" % r["arith"] if r["arith"] is not None else "?",
            "%.3f" % r["math"] if r["math"] is not None else "?",
            "%.3f" % r["string"] if r["string"] is not None else "?",
            "%.3f" % r["drift"] if r["drift"] is not None else "?"))
    lines.append("")

    # determinism cross-check vs v3 seed-42 (same learner indices -> same seeds)
    lines.append("## Determinism cross-check vs v3 seed-42 (measurement-only)")
    for name, ref in V3_REF.items():
        r = rows[name]
        d_acc = abs(r["acc"] - ref["acc"])
        lines.append("- %s: ACC %.3f (v3 %.3f, |d|=%.3f)  arith-hold %s (v3 %.3f)"
                     % (name, r["acc"], ref["acc"], d_acc,
                        "%.3f" % r["arith"] if r["arith"] is not None else "?", ref["arith"]))
    lines.append("")

    # audit trail
    lines.append("## Anchor audit trail (persisted gate entries)")
    for name in ["sccl", "sccl_anchor_lo", "sccl_anchor_hi"]:
        vals = audit_anchor(name)
        lines.append("- %s: anchor_lambda per accepted update = %s"
                     % (name, vals if vals is not None else "registry missing"))
    lines.append("")

    # pre-registered rule
    lines.append("## Pre-registered rule evaluation")
    lines.append("Rule: success for an anchor row = arith_holdout > sccl.arith_holdout "
                 "AND frontier >= sccl.frontier - 0.02.")
    verdicts = {}
    for name in ["sccl_anchor_lo", "sccl_anchor_hi"]:
        r = rows[name]
        arith_ok = (r["arith"] is not None and sccl["arith"] is not None
                    and r["arith"] > sccl["arith"])
        frontier_ok = r["frontier"] >= sccl["frontier"] - 0.02
        verdicts[name] = bool(arith_ok and frontier_ok)
        lines.append("- %s: arith %.3f vs sccl %.3f (%s); frontier %+.3f vs sccl %+.3f - 0.02 (%s) -> %s"
                     % (name,
                        r["arith"] if r["arith"] is not None else float("nan"),
                        sccl["arith"] if sccl["arith"] is not None else float("nan"),
                        "PASS" if arith_ok else "FAIL",
                        r["frontier"], sccl["frontier"],
                        "PASS" if frontier_ok else "FAIL",
                        "SUCCESS" if verdicts[name] else "FAIL"))

    both_preserve = all((rows[n]["arith"] or 0) >= 0.55 for n in ["sccl_anchor_lo", "sccl_anchor_hi"])
    both_collapse = all(rows[n]["frontier"] < sccl["frontier"] - 0.02 for n in ["sccl_anchor_lo", "sccl_anchor_hi"])
    none_lift = not any(verdicts.values()) and not both_preserve
    lines.append("")
    if any(verdicts.values()):
        lines.append("BRANCH: at least one anchor row meets the rule -> candidate mechanism; "
                     "multi-seed (43,44) required before headline claim.")
    elif both_preserve and both_collapse:
        lines.append("BRANCH: anchor preserves arith-holdout but pays a prohibitive plasticity "
                     "price -> uniform anchor too strong; next step TARGETED anchoring "
                     "(Fisher-weighted on self-certified pairs or gradient projection), gold-free.")
    elif none_lift:
        lines.append("BRANCH: anchor does NOT lift arith-holdout -> uniform L2 pull insufficient; "
                     "revisit diagnosis (per-family erosion attribution).")
    else:
        lines.append("BRANCH: mixed/partial outcome -> inspect table; decide per pre-registered spirit.")

    with open(VERDICT, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(MARKER, "w") as f:
        f.write("WATCHER-DONE\n")
    print("WATCHER-DONE verdict written to %s" % VERDICT, flush=True)
    print("\n".join(lines), flush=True)


def main():
    elapsed = 0
    while True:
        m = load_metrics()
        if m is not None:
            print("[watcher] metrics complete; waiting for runner to exit...", flush=True)
            while runner_count() > 0:
                time.sleep(20)
            break
        n = runner_count()
        if n == 0:
            print("WATCHER-RUNNER-DIED metrics missing -- run likely crashed", flush=True)
            sys.exit(2)
        if elapsed >= MAX_WAIT:
            print("WATCHER-TIMEOUT after %ds -- metrics still missing" % elapsed, flush=True)
            sys.exit(3)
        if elapsed % 1800 < INTERVAL:
            print("[watcher] waiting: elapsed=%ds runner=%s metrics=%s"
                  % (elapsed, n, os.path.exists(METRICS)), flush=True)
        time.sleep(INTERVAL)
        elapsed += INTERVAL
    analyze(m)


if __name__ == "__main__":
    main()
