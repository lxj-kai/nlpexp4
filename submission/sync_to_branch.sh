#!/usr/bin/env bash
# 构建提交副本到 submission/nlpexp4_final/（不修改项目根目录的开发代码）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python submission/build.py
rm -rf submission/nlpexp4_final
mkdir -p submission/nlpexp4_final
rsync -a submission/dist/nlpexp4_final/ submission/nlpexp4_final/

if compgen -G "submission/待填写*" > /dev/null; then
  cp submission/待填写* submission/nlpexp4_final/
fi

echo "Done: submission/nlpexp4_final/"
