#!/usr/bin/env bash
# 2Wiki 全量 main（n=5000）+ fact（最多 800）实验。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PY="${ROOT}/.venv/bin/python"
MAIN_N=5000
FACT_N=800
WORKERS=10

run_main_phase() {
  local phase="$1"
  local log="logs/exp_2wiki_${phase}_n${MAIN_N}.log"
  echo "starting ${phase} n=${MAIN_N} -> ${log}"
  "$PY" -m experiments.exp_2wiki --n "$MAIN_N" --workers "$WORKERS" --phase "$phase" >"$log" 2>&1
}

run_fact_phase() {
  local log="logs/exp_2wiki_exp1_fact_n${FACT_N}.log"
  echo "starting exp1_fact n=${FACT_N} -> ${log}"
  "$PY" -m experiments.exp_2wiki --n "$FACT_N" --workers "$WORKERS" --phase exp1_fact >"$log" 2>&1
}

run_main_phase exp1 &
run_fact_phase &
run_main_phase exp2 &
run_main_phase exp4 &
wait

"$PY" scripts/analyze_2wiki_results.py --min-n 5000
echo "done. see report/2wiki实验分析.md"
