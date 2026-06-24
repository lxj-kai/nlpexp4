#!/usr/bin/env bash
# 2Wiki n=500 全套实验：4 phase 并行，每 phase 内 10 线程并发样本。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY="${ROOT}/.venv/bin/python"
N=500
WORKERS=10

run_phase() {
  local phase="$1"
  local log="logs/exp_2wiki_${phase}_n${N}.log"
  echo "starting ${phase} -> ${log}"
  "$PY" -m experiments.exp_2wiki --n "$N" --workers "$WORKERS" --phase "$phase" >"$log" 2>&1
}

run_phase exp1 &
run_phase exp1_fact &
run_phase exp2 &
run_phase exp4 &
wait

"$PY" scripts/analyze_2wiki_results.py
echo "done. see report/2wiki实验分析.md"
