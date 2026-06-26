# 终期实验报告

本目录为**终期完整报告**（中期 RGB 实验 + 后期跨数据集验证）。

- **中期报告（勿改）**：`../report_latex/` — 自 git 保留的原版
- **终期报告（本目录）**：`main.tex` + `figures/`

## 编译

```bash
cd report_final
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex   # 第二遍更新目录
```

## 更新后期图表

```bash
python scripts/render_late_stage_figures.py
```

图表输出至 `report_final/figures/late_*.png`。
