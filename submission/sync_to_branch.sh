#!/usr/bin/env bash
# 将 staging 同步到 submission/ 根目录（保留 build.py 等工具文件）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/submission"

python build.py
STAGING="dist/staging"

# 同步交付物到 submission/ 根，不碰构建工具
rsync -a --delete \
  --exclude 'build.py' \
  --exclude 'manifest.yaml' \
  --exclude 'sync_to_branch.sh' \
  --exclude 'BUILD.md' \
  --exclude 'templates/' \
  --exclude 'dist/' \
  --exclude 'nlpexp4_final/' \
  "$STAGING/" ./

if compgen -G "待填写*" > /dev/null; then
  cp 待填写* ./ 2>/dev/null || true
fi

echo "Done. Deliverable at submission/ (src/, report_final/, experiments/, ...)"
