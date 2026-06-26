# 提交包（复制到本目录，不破坏开发环境）

本目录仅用于**复制**终期交付物，开发代码仍在仓库根目录正常运行。

## 1. 在 develop（`mhr_develop`）上构建

```bash
# 生成独立目录（不影响 src/、experiments/ 等）
python submission/build.py

# 或一步构建并复制到 submission/nlpexp4_final/
bash submission/sync_to_branch.sh
```

- 构建缓存：`submission/dist/nlpexp4_final/`（已 gitignore）
- 待推送副本：`submission/nlpexp4_final/`（同步到 `submission` 分支时使用）

zip 压缩请本地人工完成，脚本不生成 zip。

## 2. 更新 submission 分支

```bash
bash submission/sync_to_branch.sh

git checkout submission
# 以 develop 为基，保留 submission/nlpexp4_final/ 副本
git checkout mhr_develop -- .
git add submission/nlpexp4_final/ submission/README.md
git commit -m "Update submission package copy"
git push origin submission

git checkout mhr_develop
```

`submission` 分支结构与 develop 相同，**额外**在 `submission/nlpexp4_final/` 下有完整提交副本。老师可只关注该子目录。

**终期报告路径**：`submission/nlpexp4_final/report_final/main.pdf`（不是 `report_latex/`）。

## 3. 目录说明

```text
submission/
├── build.py           # 从仓库根目录复制文件到 dist/
├── manifest.yaml      # 包含/排除规则
├── sync_to_branch.sh  # 构建 + 复制到 nlpexp4_final/
├── templates/         # 输出包 README 模板
├── dist/              # 构建输出（gitignore）
└── nlpexp4_final/     # 提交副本（仅 submission 分支跟踪）
```
