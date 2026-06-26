#!/usr/bin/env bash
# 构建提交包副本，并推送到 submission 分支（不破坏 develop 工作区）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEV_BRANCH="${DEV_BRANCH:-mhr_develop}"
SUB_BRANCH="${SUB_BRANCH:-submission}"
CUR="$(git branch --show-current)"

echo "==> 1/4 构建 submission/nlpexp4_final/"
bash "$ROOT/submission/sync_to_branch.sh"

# 课程表单一并放入提交副本
if compgen -G "$ROOT/submission/待填写*" > /dev/null; then
  cp "$ROOT/submission/待填写"* "$ROOT/submission/nlpexp4_final/"
fi

echo "==> 2/4 切换到 $SUB_BRANCH"
git checkout "$SUB_BRANCH"
git merge "$DEV_BRANCH" -m "Merge $DEV_BRANCH into $SUB_BRANCH" || true

echo "==> 3/4 更新 submission/nlpexp4_final/（保留分支上已有大文件）"
mkdir -p submission/nlpexp4_final
rsync -a "$ROOT/submission/nlpexp4_final/" submission/nlpexp4_final/

git add submission/nlpexp4_final/ submission/README.md submission/manifest.yaml \
  submission/build.py submission/sync_to_branch.sh submission/templates/ \
  submission/.gitignore README.md .gitignore 2>/dev/null || true
git add -f submission/nlpexp4_final/

if git diff --cached --quiet; then
  echo "No changes to commit on $SUB_BRANCH."
else
  git commit -m "Update submission package copy under submission/nlpexp4_final/"
fi

echo "==> 4/4 推送到 origin/$SUB_BRANCH"
git push origin "$SUB_BRANCH"

git checkout "$CUR"
echo "Done. Submission package: submission/nlpexp4_final/ on branch $SUB_BRANCH"
