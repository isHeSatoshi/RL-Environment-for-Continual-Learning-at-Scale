"""Detached watcher: wait for the multi-seed v3 run (seeds 43,44) to finish,
then aggregate 42/43/44 and emit paper/results_v3.tex.

Hardened so it ALWAYS returns control to the agent:
  * success  -> seed-44 metrics valid, waits for the run_seeds orchestrator to
                exit (it runs its own 2-seed aggregate on exit, which we must
                not race), then re-aggregates with all three seeds and writes
                the registry + a WATCHER-DONE marker.
  * crash    -> orchestrator process gone but seed-44 metrics missing -> exit 2.
  * timeout  -> wall-clock cap exceeded -> exit 3.
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

M44 = "runs/sccl_v3_s44/metrics.json"
MARKER = "runs/sccl_v3_seeds_watchdone.txt"
MAX_WAIT = 6 * 3600     # hard wall-clock cap
INTERVAL = 120


def orch_count() -> int:
    """Number of live run_seeds.py orchestrator processes. -1 = unknown."""
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*run_seeds.py*' } | "
           "Measure-Object | Select-Object -ExpandProperty Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=60)
        s = out.stdout.strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return -1


def metrics_valid(path: str) -> bool:
    try:
        m = json.load(open(path))
        return "learners" in m and len(m["learners"]) >= 5
    except Exception:
        return False


elapsed = 0
while True:
    if metrics_valid(M44):
        print("[watcher] seed-44 metrics valid; waiting for orchestrator to exit...", flush=True)
        while True:
            if orch_count() == 0:
                break
            time.sleep(20)
        print("[watcher] orchestrator exited; aggregating 42/43/44", flush=True)
        break

    n = orch_count()
    if n == 0:
        print("WATCHER-ORCHESTRATOR-DIED seed-44 metrics missing -- run likely crashed", flush=True)
        sys.exit(2)
    if elapsed >= MAX_WAIT:
        print("WATCHER-TIMEOUT after %ds -- seed-44 metrics still missing" % elapsed, flush=True)
        sys.exit(3)
    time.sleep(INTERVAL)
    elapsed += INTERVAL

r = subprocess.run([sys.executable, "scripts/run_seeds.py", "--config",
                    "configs/sccl_v3.json", "--seeds", "42,43,44", "--aggregate-only"])
if r.returncode != 0:
    print("WATCHER-AGG-FAIL", flush=True)
    sys.exit(1)

r = subprocess.run([sys.executable, "-m", "gcl.report", "--aggregate",
                    "runs/sccl_v3_seeds", "--out", "paper/results_v3.tex",
                    "--prefix", "vthree"])
if r.returncode != 0:
    print("WATCHER-REPORT-FAIL", flush=True)
    sys.exit(1)

with open(MARKER, "w") as f:
    f.write("WATCHER-DONE\n")
print("WATCHER-DONE results_v3.tex generated from 3-seed aggregate", flush=True)
