# 提交包构建说明

本目录用于在**开发分支**上生成独立终期提交目录。打包 zip 上传由人工完成。

## 构建

在仓库根目录执行：

```bash
python submission/build.py
python submission/build.py --dry-run   # 仅预览
```

输出：`submission/dist/nlpexp4_final/`（独立目录，可人工压缩后提交给老师）

## 提交分支

终期交付物已单独放在 **`submission` 分支**（仓库根目录即为提交包内容）。开发分支 `mhr_develop` 不含本目录。

## 目录说明

```text
submission/
├── build.py              # 构建脚本（仅开发用）
├── manifest.yaml         # 包含/排除规则
├── templates/
│   ├── README.md         # 写入输出包根目录的 README 模板
│   └── package.gitignore
└── dist/                 # 构建产物（gitignore）
    └── nlpexp4_final/
```

修改 `manifest.yaml` 或 `templates/README.md` 后重新运行 `build.py`，再同步到 `submission` 分支。
