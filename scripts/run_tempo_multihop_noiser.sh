#!/usr/bin/env bash
# TEMPO / MultiHop-RAG 矫正对比 + NoiserBench 七类噪音 正式实验
#
# 用法:
#   bash scripts/run_tempo_multihop_noiser.sh
#   bash scripts/run_tempo_multihop_noiser.sh --dry --n 2
#   bash scripts/run_tempo_multihop_noiser.sh --n 50 --noiser-only
#   nohup bash scripts/run_tempo_multihop_noiser.sh > logs/run_tempo_multihop_noiser.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

N=50
DRY=""
CORRECTION_ONLY=""
NOISER_ONLY=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --dry) DRY="--dry"; shift ;;
    --correction-only) CORRECTION_ONLY=1; shift ;;
    --noiser-only) NOISER_ONLY=1; shift ;;
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

TS=$(date +%Y%m%d_%H%M%S)
MASTER="logs/run_tempo_multihop_noiser_${TS}.log"

run_correction() {
  local log="logs/exp_correction_n${N}_${TS}.log"
  echo ""
  echo ">>> TEMPO + MultiHop-RAG correction n=$N  log=$log"
  local -a cmd=(
    python3 -m experiments.exp_dataset_correction
    --n "$N"
    --methods naive,prompt,confidence,voting
    --ratios 0,0.5,0.75
  )
  [[ -n "$DRY" ]] && cmd+=("$DRY")
  ((${#EXTRA[@]})) && cmd+=("${EXTRA[@]}")
  "${cmd[@]}" 2>&1 | tee "$log"
}

run_noiser() {
  local log="logs/exp_noiser_n${N}_${TS}.log"
  echo ""
  echo ">>> NoiserBench 7-type exp1+exp2 n=$N  log=$log"
  local -a cmd=(
    python3 -m experiments.exp_noiser_bench
    --n "$N"
    --subset hotpotqa
    --phase all
    --methods naive,prompt,confidence,voting
  )
  [[ -n "$DRY" ]] && cmd+=("$DRY")
  ((${#EXTRA[@]})) && cmd+=("${EXTRA[@]}")
  "${cmd[@]}" 2>&1 | tee "$log"
}

{
  echo "started at $(date -Iseconds)"
  echo "n=$N dry=${DRY:-no}"

  if [[ -z "$NOISER_ONLY" ]]; then
    run_correction
  fi
  if [[ -z "$CORRECTION_ONLY" ]]; then
    run_noiser
  fi

  echo "finished at $(date -Iseconds)"
} 2>&1 | tee "$MASTER"

echo ""
echo "=== done ==="
echo "master log : $MASTER"
echo "correction : experiments/results/exp_correction_*_n${N}_*.json"
echo "noiser     : experiments/results/exp_noiser_*_n${N}_*.json"
echo "figures    : figures/exp_correction_*  figures/exp_noiser_*"
