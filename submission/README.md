# submission 目录

- **`nlpexp4_final/`** — 终期提交副本（给老师），README 在其中的根目录
- **`build.py` / `manifest.yaml`** — 从项目根目录复制文件重建副本
- **`dist/`** — 构建缓存（gitignore）

```bash
# 重建提交副本（不修改 src/、experiments/ 等开发目录）
bash submission/sync_to_branch.sh

# 或分步
python submission/build.py
rsync -a submission/dist/nlpexp4_final/ submission/nlpexp4_final/
```

zip 压缩请本地人工完成。老师使用 `submission/nlpexp4_final/` 即可，与仓库其余部分无依赖。
