# 终期实验报告

本目录为**终期完整报告**（2Wiki / MobileMem / Cmedqa 主实验 + Judge 评分 + 统一流水线）。

- **历史快照**：`../report_latex/` — 早期探索版本，不作为终期正文依据
- **终期报告（本目录）**：`main.tex` + `figures/`

## 编译

```bash
cd report_final
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex   # 第二遍更新目录
```

## 更新扩展数据集图表

```bash
python scripts/render_late_stage_figures.py
```

图表输出至 `report_final/figures/late_*.png`。
