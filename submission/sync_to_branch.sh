#!/usr/bin/env bash
# 在 develop 上：构建提交包并复制到 submission/nlpexp4_final/，供 submission 分支推送。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python submission/build.py
rm -rf submission/nlpexp4_final
mkdir -p submission/nlpexp4_final
rsync -a submission/dist/nlpexp4_final/ submission/nlpexp4_final/

echo "Done: submission/nlpexp4_final/"
echo "Next: git checkout submission && git checkout mhr_develop -- . ':!submission/nlpexp4_final' && ..."
