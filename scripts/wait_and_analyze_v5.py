"""Detached watcher + analyzer for the SCCL v5 2x2 factorial (runs/sccl_v5).

Waits for the runner to finish (metrics.json with all 6 learners), then
evaluates the PRE-REGISTERED v5 decision rules (RESEARCH_NOTES_v5.md) and
writes runs/sccl_v5_verdict.md. Never touches gold in any decision here:
gold fields below are measurement-only determinism cross-checks.

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

METRICS = "runs/sccl_v5/metrics.json"
VERDICT = "runs/sccl_v5_verdict.md"
MARKER = "runs/sccl_v5_watchdone.txt"
MAX_WAIT = 8 * 3600
INTERVAL = 120
LEARNERS = ["frozen", "sccl", "sccl_strat", "sccl_anchor_lo",
            "sccl_anchor_strat", "vsr_nogold"]

# seed-42 reference rows that sccl/frozen must bit-reproduce (v3/v4 ladders had
# sccl at learner index 1 -> seed 43, frozen at index 0). Measurement-only.
REF = {
    "frozen": {"acc": 0.613, "arith": 0.600},
    "sccl":   {"acc": 0.575, "arith": 0.400},
}


def runner_count() -> int:
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*sccl_v5.json*' } | "
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


def audit(learner):
    """Persisted gate audit trail: set of (stratified, anchor_lambda)."""
    path = f"runs/sccl_v5/adapters_{learner}/registry.json"
    try:
        d = json.load(open(path))
        metas = d if isinstance(d, list) else d.get("metas", [])
        return sorted({((mm.get("gate") or {}).get("stratified"),
                        (mm.get("gate") or {}).get("anchor_lambda")) for mm in metas})
    except Exception:
        return None


def analyze(m):
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
    L = []
    L.append("# SCCL v5 verdict (seed 42, 2x2 factorial anchor x coverage, runs/sccl_v5)")
    L.append("")
    L.append("| learner | anchor | strat | ACC | BWT | forget | frontier | upd/rbk | arith-hold | math | string | drift |")
    L.append("|---------|:---:|:---:|-----|-----|--------|----------|---------|-----------|------|--------|-------|")
    fac = {"frozen": ("-", "-"), "sccl": ("-", "-"), "sccl_strat": ("-", "Y"),
           "sccl_anchor_lo": ("0.1", "-"), "sccl_anchor_strat": ("0.1", "Y"),
           "vsr_nogold": ("-", "-")}
    for name in LEARNERS:
        r = rows[name]
        a, s = fac[name]
        L.append("| %s | %s | %s | %.3f | %+.3f | %.3f | %+.3f | %d/%d | %s | %s | %s | %s |" % (
            name, a, s, r["acc"], r["bwt"], r["forgetting"], r["frontier"],
            r["updates"], r["rollbacks"],
            "%.3f" % r["arith"] if r["arith"] is not None else "?",
            "%.3f" % r["math"] if r["math"] is not None else "?",
            "%.3f" % r["string"] if r["string"] is not None else "?",
            "%.3f" % r["drift"] if r["drift"] is not None else "?"))
    L.append("")

    L.append("## Determinism cross-check vs v3/v4 seed-42 (measurement-only)")
    for name, ref in REF.items():
        r = rows[name]
        L.append("- %s: ACC %.3f (ref %.3f, |d|=%.3f)  arith-hold %s (ref %.3f)"
                 % (name, r["acc"], ref["acc"], abs(r["acc"] - ref["acc"]),
                    "%.3f" % r["arith"] if r["arith"] is not None else "?", ref["arith"]))
    L.append("")

    L.append("## Audit trail (persisted gate entries: (stratified, anchor_lambda))")
    for name in ["sccl", "sccl_strat", "sccl_anchor_lo", "sccl_anchor_strat"]:
        L.append("- %s: %s" % (name, audit(name) if audit(name) is not None else "registry missing"))
    L.append("")

    L.append("## Pre-registered rule evaluation")
    L.append("H1 coverage: sccl_strat.arith > sccl.arith AND frontier >= sccl.frontier - 0.02")
    L.append("H2 composition: sccl_anchor_strat.arith > sccl.arith AND frontier >= sccl.frontier - 0.02")
    L.append("BREAKTHROUGH: sccl_anchor_strat.arith >= 0.55 AND frontier >= sccl.frontier - 0.02")
    L.append("")

    def rule(name):
        r = rows[name]
        arith_ok = (r["arith"] is not None and sccl["arith"] is not None
                    and r["arith"] > sccl["arith"])
        frontier_ok = r["frontier"] >= sccl["frontier"] - 0.02
        return arith_ok, frontier_ok

    h1a, h1f = rule("sccl_strat")
    h2a, h2f = rule("sccl_anchor_strat")
    bt_arith = (rows["sccl_anchor_strat"]["arith"] is not None
                 and rows["sccl_anchor_strat"]["arith"] >= 0.55)
    bt = bt_arith and h2f
    L.append("- H1 sccl_strat: arith %.3f vs sccl %.3f (%s); frontier %+.3f vs %+.3f-0.02 (%s) -> %s"
             % (rows["sccl_strat"]["arith"], sccl["arith"], "PASS" if h1a else "FAIL",
                rows["sccl_strat"]["frontier"], sccl["frontier"], "PASS" if h1f else "FAIL",
                "SUPPORTED" if (h1a and h1f) else "NOT-SUPPORTED"))
    L.append("- H2 sccl_anchor_strat: arith %.3f vs sccl %.3f (%s); frontier %+.3f vs %+.3f-0.02 (%s) -> %s"
             % (rows["sccl_anchor_strat"]["arith"], sccl["arith"], "PASS" if h2a else "FAIL",
                rows["sccl_anchor_strat"]["frontier"], sccl["frontier"], "PASS" if h2f else "FAIL",
                "SUPPORTED" if (h2a and h2f) else "NOT-SUPPORTED"))
    L.append("- BREAKTHROUGH sccl_anchor_strat: arith %.3f >= 0.55 (%s) AND frontier ok (%s) -> %s"
             % (rows["sccl_anchor_strat"]["arith"], "PASS" if bt_arith else "FAIL",
                "PASS" if h2f else "FAIL", "YES" if bt else "no"))
    L.append("")

    # main effects + interaction summary
    base = sccl["arith"] or 0
    strat_eff = (rows["sccl_strat"]["arith"] or 0) - base
    anchor_eff = (rows["sccl_anchor_lo"]["arith"] or 0) - base
    both_eff = (rows["sccl_anchor_strat"]["arith"] or 0) - base
    L.append("## Main effects on arith-holdout (vs sccl baseline %.3f)" % base)
    L.append("- coverage alone (sccl_strat - sccl):      %+.3f" % strat_eff)
    L.append("- anchor alone   (sccl_anchor_lo - sccl):  %+.3f" % anchor_eff)
    L.append("- both           (sccl_anchor_strat - sccl): %+.3f" % both_eff)
    inter = both_eff - (strat_eff + anchor_eff)
    L.append("- interaction (both - [strat + anchor]):   %+.3f  (%s)"
             % (inter, "super-additive" if inter > 0.02 else
                ("additive" if abs(inter) <= 0.02 else "sub-additive")))
    L.append("")

    if bt:
        L.append("BRANCH: BREAKTHROUGH candidate (anchor+coverage hits arith>=0.55 without "
                 "frontier loss) -> multi-seed (43,44) REQUIRED before headline claim.")
    elif (h2a and h2f):
        L.append("BRANCH: composition lifts arith above sccl (but < 0.55) -> partial win; "
                 "multi-seed to confirm, then consider stronger coverage (capability probes).")
    elif (h1a and h1f):
        L.append("BRANCH: coverage ALONE lifts arith; anchor adds nothing for arith -> report "
                 "anchor as stream-only helper; multi-seed the coverage row.")
    else:
        L.append("BRANCH: neither coverage nor anchor lifts arith -> (G) generalization gap "
                 "dominates; next step targeted SUBSPACE protection (Fisher-weighted anchor on "
                 "arith directions / gradient projection), still gold-free. Informative negative.")

    with open(VERDICT, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(MARKER, "w") as f:
        f.write("WATCHER-DONE\n")
    print("WATCHER-DONE verdict written to %s" % VERDICT, flush=True)
    print("\n".join(L), flush=True)


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
