#!/usr/bin/env bash
# 全数据集 n=50 批量评测：噪音梯度 × 矫正方法 → JSON + 图表
#
# 模型（读 .env）：
#   生成  LMSTUDIO_MODEL=qwen/qwen3-4b-2507
#   判别  DEEPSEEK_MODEL=deepseek-chat
#
# 用法：
#   bash scripts/run_all_datasets_n50.sh
#   bash scripts/run_all_datasets_n50.sh --dry          # 离线接线检查
#   bash scripts/run_all_datasets_n50.sh --n 10         # 快速试跑
#   nohup bash scripts/run_all_datasets_n50.sh > logs/run_all_datasets_n50.log 2>&1 &

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
print("gen base  :", CONFIG.api_base)
print("judge     :", CONFIG.judge_model)
print("judge base:", CONFIG.judge_api_base)
PY

METHODS="naive,prompt,confidence,voting"
RATIOS="0,0.5,0.75"
TS=$(date +%Y%m%d_%H%M%S)
MASTER="logs/run_all_datasets_n50_${TS}.log"

# dataset language subset
RUNS=(
  "rgb|zh|main"
  "2wiki|en|main"
  "cmedqa|zh|main"
  "miriad|en|main"
  "bright|en|main"
  "multihop_rag|en|main"
  "tempo|en|main"
  "noiser_bench|en|hotpotqa"
  "mobilemem|zh|calc"
)

run_one() {
  local ds="$1" lang="$2" sub="$3"
  local tag="${ds}_${lang}_${sub}_n${N}"
  local log="logs/smoke_${tag}_${TS}.log"
  echo ""
  echo ">>> [$ds] lang=$lang subset=$sub n=$N  log=$log"
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
  echo "n=$N methods=$METHODS ratios=$RATIOS dry=${DRY:-no}"
  for entry in "${RUNS[@]}"; do
    IFS='|' read -r ds lang sub <<< "$entry"
    run_one "$ds" "$lang" "$sub"
  done
  echo "finished at $(date -Iseconds)"
} 2>&1 | tee "$MASTER"

echo ""
echo "=== done ==="
echo "master log : $MASTER"
echo "results    : experiments/results/smoke_*_n${N}_*.json"
echo "figures    : figures/smoke_*_n${N}_*/"
