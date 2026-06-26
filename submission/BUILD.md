# 构建说明（开发用）

老师使用的代码与报告在 **`submission/` 根目录**：

```text
submission/
├── README.md        # 使用说明
├── src/             # 全部源码
├── experiments/     # 实验脚本 + results/
├── report_final/    # 终期报告（main.pdf）
├── backend/ frontend/ data/ figures/ scripts/ tests/
├── build.py         # 以下为本机构建工具
└── ...
```

```bash
bash submission/sync_to_branch.sh
git add submission/
git commit -m "Update submission deliverable"
git push origin mhr_develop
```

zip 给老师：进入 `submission/`，压缩除 `build.py`、`manifest.yaml`、`dist/`、`templates/`、`BUILD.md` 以外的内容；或直接压缩整个 `submission/` 目录。
