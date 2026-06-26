# 提交包构建说明

本目录用于生成**独立终期提交包**，供打包上传给老师。输出物与开发仓库解耦，解压后可单独使用。

## 快速构建

在**开发仓库根目录**执行：

```bash
# 生成 submission/dist/nlpexp4_final/
python submission/build.py

# 同时生成 zip（推荐上传）
python submission/build.py --zip

# 预览将复制多少文件（不写入磁盘）
python submission/build.py --dry-run
```

上传给老师：`submission/dist/nlpexp4_final.zip`

## 目录说明

```text
submission/
├── build.py              # 构建脚本
├── manifest.yaml         # 包含/排除规则
├── templates/
│   ├── README.md         # 写入输出包根目录的 README 模板
│   └── package.gitignore
└── dist/                 # 构建产物（gitignore，本地生成）
    ├── nlpexp4_final/    # 独立提交目录
    └── nlpexp4_final.zip
```

## 自定义

- 修改 **`manifest.yaml`** 的 `include` / `exclude_globs` 调整打包范围
- 修改 **`templates/README.md`** 更新提交包内说明文档
- 输出包内会额外生成 **`SUBMISSION_MANIFEST.json`**（全文件 SHA256 清单）

## 设计原则

1. **独立**：输出目录不含对开发路径的引用
2. **干净**：排除 `__pycache__`、`.env`、`node_modules`、LaTeX 中间文件、GB 级 raw 数据
3. **完整**：含源码、全部 JSON 实验结果、图表、终期 PDF 报告、演示前后端

## 体积参考

完整构建约 **900 MiB**（主要为 `experiments/results/` 与 `data/rgb/`）。若需精简，可在 `manifest.yaml` 中注释掉部分 `experiments/results` 子目录。
