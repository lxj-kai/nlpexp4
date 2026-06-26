#!/usr/bin/env bash
# mobilemem + bright + 2wiki，各 n=50：噪音梯度 × 矫正 → JSON + 图表
#
# 用法：
#   bash scripts/run_three_datasets_n50.sh
#   nohup bash scripts/run_three_datasets_n50.sh > logs/run_three_datasets_n50.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

N=50
DRY=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --dry) DRY="--dry"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

mkdir -p logs figures experiments/results

echo "=== config ==="
python3 - <<'PY'
from src.config import CONFIG
print("gen model :", CONFIG.model)
print("judge     :", CONFIG.judge_model)
PY

METHODS="naive,prompt,confidence,voting"
RATIOS="0,0.5,0.75"
TS=$(date +%Y%m%d_%H%M%S)
MASTER="logs/run_three_datasets_n50_${TS}.log"

RUNS=(
  "mobilemem|zh|calc"
  "bright|en|main"
  "2wiki|en|main"
)

run_one() {
  local ds="$1" lang="$2" sub="$3"
  local tag="${ds}_${lang}_${sub}_n${N}"
  local log="logs/smoke_${tag}_${TS}.log"
  echo ""
  echo ">>> [$ds] lang=$lang subset=$sub n=$N"
  python3 -m src.smoke_test \
    --n "$N" \
    --dataset "$ds" \
    --language "$lang" \
    --subset "$sub" \
    --methods "$METHODS" \
    --ratios "$RATIOS" \
    --noise-type semantic \
    --noise-position interleave \
    --figures-dir "figures/smoke_${tag}_${TS}" \
    $DRY \
    "${EXTRA[@]}" \
    2>&1 | tee "$log"
}

{
  echo "started at $(date -Iseconds)"
  echo "datasets: mobilemem bright 2wiki | n=$N"
  for entry in "${RUNS[@]}"; do
    IFS='|' read -r ds lang sub <<< "$entry"
    run_one "$ds" "$lang" "$sub"
  done
  echo "finished at $(date -Iseconds)"
} 2>&1 | tee "$MASTER"

echo ""
echo "master log : $MASTER"
echo "figures    : figures/smoke_{mobilemem,bright,2wiki}_*"
