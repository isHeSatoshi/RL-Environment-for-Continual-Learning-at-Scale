#!/usr/bin/env bash
# One-shot paper build: regenerate result registries from completed runs,
# then compile main.tex with TinyTeX. Missing runs are skipped so the paper
# always builds from whatever has finished.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/c/Users/Aditya/AppData/Roaming/TinyTeX/bin/windows:$PATH"

if [ -f runs/sccl_main/metrics.json ]; then
  python -m gcl.report --run runs/sccl_main --out paper/results.tex
else
  echo "WARN: runs/sccl_main/metrics.json missing — keeping existing paper/results.tex"
fi

if [ -f runs/sccl_v2/metrics.json ]; then
  python -m gcl.report --run runs/sccl_v2 --out paper/results_v2.tex --prefix vtwo
else
  echo "WARN: runs/sccl_v2/metrics.json missing — v2 table falls back to placeholders"
fi

cd paper
pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > compile.log 2>&1
if grep -qE "^!" compile.log; then
  echo "LaTeX errors:"; grep -E "^!" compile.log | head; exit 1
fi
echo "OK: paper/main.pdf built"
