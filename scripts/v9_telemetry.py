"""Post-hoc gold telemetry for the v9 dose ladder (Branch G).

Thin wrapper over scripts/v8_telemetry.py with the v9 rows substituted:
the M1 gen-witness layer, manufacture/retirement ledger (src is a LIST since
the G-lane change), gate stats, vault lanes, erosion timelines. MEASUREMENT
ONLY — gold scores were logged at run time and never entered any
accept/reject decision.

Usage: python scripts/v9_telemetry.py [--run runs/sccl_v9_s42]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v8_telemetry as T  # noqa: E402

T.LEARNERS = ["sccl", "sccl_capprobe", "sccl_capprobe_strat",
              "sccl_gen2_half", "sccl_gen2_majority"]
T.THETA_ROWS = {"sccl_gen2_half": 0.5, "sccl_gen2_majority": 2 / 3}
T.GEN_ROWS = {"sccl_gen2_half", "sccl_gen2_majority"}
T.ANCHOR_ROWS = {"sccl_gen2_half", "sccl_gen2_majority"}

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["--run", "runs/sccl_v9_s42"]
    T.main()
