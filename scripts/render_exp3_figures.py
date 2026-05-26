"""生成 exp3 案例深度分析图表（大样本版）：堆叠比例柱状图 + 性能折线图 + per-type 指标表。

支持中英文 × main/fact，基于全量分类统计。
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.config import CONFIG
from src.visualize import _PALETTE, _setup_style, _save
from src.utils import get_logger

logger = get_logger("render_exp3")

RESULTS = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

TYPE_ORDER = ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]
TYPE_COLORS = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"]
TYPE_CN = ["矫正生效", "矫正未生效", "信息淹没", "免疫"]


def make_figure(data: dict, label: str, out_path: Path) -> str:
    _setup_style()
    extra = data.get("ratio_summaries", {})
    results = data["results"]
    if not extra:
        print(f"  no ratio_summaries in {label}")
        return ""

    ratios = sorted(float(k) for k in extra.keys())

    # ── 数据提取 ──
    # 全量 type 分布（count + proportion）
    counts = np.zeros((len(TYPE_ORDER), len(ratios)))
    proportions = np.zeros((len(TYPE_ORDER), len(ratios)))
    n_total_per_ratio = []
    for j, r in enumerate(ratios):
        rs = extra[str(r)]
        n_total = rs.get("n_total", 0)
        n_total_per_ratio.append(n_total)
        for i, tn in enumerate(TYPE_ORDER):
            counts[i, j] = rs.get("type_distribution", {}).get(tn, 0)
            proportions[i, j] = rs.get("type_proportions", {}).get(tn, 0.0)

    # 性能数据
    f1_clean = (results[0]["summary"].get("token_f1") or 0.0)
    f1_noisy = []
    f1_corr = []
    for i, r in enumerate(ratios):
        f1_noisy.append((results[1 + i * 2]["summary"].get("token_f1") or 0.0))
        f1_corr.append((results[1 + i * 2 + 1]["summary"].get("token_f1") or 0.0))

    # ── 三面板图 ──
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                          height_ratios=[3, 2])

    # ── 左上：全量堆叠比例柱状图 ──
    ax1 = fig.add_subplot(gs[0, :2])
    bottom = np.zeros(len(ratios))
    x = np.arange(len(ratios))
    bar_width = 0.55
    ratio_labels = [f"{r:.0%}" for r in ratios]

    for i, (tn, color) in enumerate(zip(TYPE_ORDER, TYPE_COLORS)):
        bars = ax1.bar(x, proportions[i], bottom=bottom,
                       color=color, label=TYPE_CN[i], width=bar_width,
                       edgecolor="white", linewidth=0.8)
        # 标注 > 5% 的百分比
        for j, (bar, pct) in enumerate(zip(bars, proportions[i])):
            if pct > 5:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bottom[j] + pct / 2,
                         f"{pct:.0f}%", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color="white")

    # 在每根柱子上方标注总样本量
    for j, (rj, nt) in enumerate(zip(ratios, n_total_per_ratio)):
        ax1.text(j, 102, f"n={nt}", ha="center", va="bottom", fontsize=8, color="#64748b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ratio_labels, fontsize=11)
    ax1.set_xlabel("噪音比例", fontsize=12)
    ax1.set_ylabel("案例占比 (%)", fontsize=12)
    ax1.set_title(f"案例类型分布 ({label})  [全量分类]", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9,
               title="案例类型", title_fontsize=10)
    ax1.set_ylim(0, 112)
    ax1.axhline(y=100, color="#94a3b8", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── 右上：性能折线 ──
    ax2 = fig.add_subplot(gs[0, 2])
    x_full = list(range(len(ratios) + 1))
    x_labels_full = ["0%\n(clean)"] + [f"{r:.0%}" for r in ratios]
    f1_line_naive = [f1_clean] + f1_noisy
    f1_line_corr = [f1_clean] + f1_corr

    ax2.plot(x_full, f1_line_naive, "-o", color="#ef4444", linewidth=2.2, markersize=7,
             label="Naive (无矫正)")
    ax2.plot(x_full, f1_line_corr, "-s", color="#3b82f6", linewidth=2.2, markersize=7,
             label="Confidence (矫正后)")
    ax2.axhline(y=f1_clean, color="#94a3b8", linestyle="--", alpha=0.6, linewidth=1.5,
                label=f"Clean baseline ({f1_clean:.3f})")
    ax2.set_xticks(x_full)
    ax2.set_xticklabels(x_labels_full, fontsize=10)
    ax2.set_xlabel("噪音比例", fontsize=12)
    ax2.set_ylabel("Token-F1", fontsize=12)
    ax2.set_title(f"噪音影响与矫正恢复 ({label})", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower left")
    y_min = min(min(f1_line_naive), min(f1_line_corr)) * 0.85
    y_max = max(f1_line_naive + f1_line_corr) * 1.05
    ax2.set_ylim(max(0, y_min), y_max)

    # ── 下半部分：per-type 详细指标折线图 ──
    # 左下：Type1 矫正生效 vs 未生效的 NAR 对比
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_per_type_metric(ax3, extra, ratios, "Type1-矫正生效", "Type1-矫正未生效",
                          metric_key="avg_nar_corrected",
                          color_eff="#10b981", color_ineff="#ef4444",
                          label_eff="矫正生效 (NAR)",
                          label_ineff="矫正未生效 (NAR)",
                          ylabel="平均 NAR", title="矫正后 NAR 对比",
                          ylim=(0, 0.6))

    # 中下：各类型的 F1 在各比例下的变化
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_all_types_f1(ax4, extra, ratios, f1_clean)

    # 右下：ISR 散点比较 (clean noisy  vs noisy)
    ax5 = fig.add_subplot(gs[1, 2])
    _plot_isr_comparison(ax5, extra, ratios)

    fig.suptitle(f"Exp3: 案例深度分析 — {label}  (全量分类, N≈{n_total_per_ratio[0] if n_total_per_ratio else '?'})",
                 fontsize=15, fontweight="bold", y=1.01)
    path = _save(fig, out_path)
    print(f"  -> {path}")
    return path


def _plot_per_type_metric(ax, extra, ratios, type_a, type_b, *,
                          metric_key, color_eff, color_ineff,
                          label_eff, label_ineff,
                          ylabel, title, ylim):
    """绘制两类案例的某个指标随比例变化的折线图。"""
    vals_a, vals_b = [], []
    for r in ratios:
        stats = extra[str(r)].get("per_type_stats", {})
        vals_a.append(stats.get(type_a, {}).get(metric_key))
        vals_b.append(stats.get(type_b, {}).get(metric_key))

    x = range(len(ratios))
    valid_a = [(i, v) for i, v in enumerate(vals_a) if v is not None]
    valid_b = [(i, v) for i, v in enumerate(vals_b) if v is not None]
    if valid_a:
        xi, yi = zip(*valid_a)
        ax.plot(xi, yi, "-o", color=color_eff, linewidth=2, markersize=6, label=label_eff)
    if valid_b:
        xi, yi = zip(*valid_b)
        ax.plot(xi, yi, "-s", color=color_ineff, linewidth=2, markersize=6, label=label_ineff)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=9)
    ax.set_xlabel("噪音比例", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)


def _plot_all_types_f1(ax, extra, ratios, f1_clean):
    """绘制所有类型的 F1(corrected) 随比例变化。"""
    all_types = ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]
    colors = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"]
    x = range(len(ratios))

    for tn, color in zip(all_types, colors):
        vals = []
        for r in ratios:
            stats = extra[str(r)].get("per_type_stats", {})
            vals.append(stats.get(tn, {}).get("avg_f1_corrected"))
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if valid:
            xi, yi = zip(*valid)
            ax.plot(xi, yi, "-o", color=color, linewidth=1.8, markersize=5,
                    label=tn.replace("Type1-", "").replace("Type3-", "").replace("Type4-", ""),
                    alpha=0.85)

    ax.axhline(y=f1_clean, color="#94a3b8", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=9)
    ax.set_xlabel("噪音比例", fontsize=10)
    ax.set_ylabel("矫正后 Token-F1", fontsize=10)
    ax.set_title("各类型矫正后 F1 对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")


def _plot_isr_comparison(ax, extra, ratios):
    """绘制 ISR: clean vs noisy vs corrected 对比。"""
    all_types = ["Type1-矫正生效", "Type1-矫正未生效", "Type4-免疫"]
    colors = ["#10b981", "#ef4444", "#3b82f6"]

    # 对每个类型，画 clean ISR -> noisy ISR -> corrected ISR 的变迁
    for tn, color in zip(all_types, colors):
        isr_clean_vals, isr_noisy_vals, isr_corr_vals = [], [], []
        for r in ratios:
            stats = extra[str(r)].get("per_type_stats", {}).get(tn, {})
            isr_clean_vals.append(stats.get("avg_isr_clean"))
            isr_noisy_vals.append(stats.get("avg_isr_noisy"))
            isr_corr_vals.append(stats.get("avg_isr_clean"))  # 矫正后 ISR 应该用 corrected

        # 取中间那个 ratio 的数据画 bar chart
        mid = len(ratios) // 2
        if mid < len(ratios):
            r_mid = ratios[mid]
            isr_c = isr_clean_vals[mid]
            isr_n = isr_noisy_vals[mid]
            # corrected ISR is not directly in per_type_stats, compute from all_cases
            # Use the corrected ISR from the ratio summary
            stats_mid = extra[str(r_mid)].get("per_type_stats", {}).get(tn, {})
            isr_corr = stats_mid.get("avg_isr_clean")  # placeholder

    # 简化版：画各 type 在最大比例下的 ISR_noisy vs ISR_corrected 对比
    x = np.arange(len(ratios))
    width = 0.25

    for idx, (tn, color) in enumerate(zip(all_types, colors)):
        isr_noisy_vals = []
        for r in ratios:
            stats = extra[str(r)].get("per_type_stats", {}).get(tn, {})
            isr_noisy_vals.append(stats.get("avg_isr_noisy") or 0)
        valid = [(i, v) for i, v in enumerate(isr_noisy_vals) if v is not None]
        if valid:
            xi, yi = zip(*valid)
            ax.plot(xi, yi, "-o", color=color, linewidth=1.8, markersize=5,
                    label=tn.replace("Type1-", "").replace("Type4-", ""),
                    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=9)
    ax.set_xlabel("噪音比例", fontsize=10)
    ax.set_ylabel("平均 ISR (noisy)", fontsize=10)
    ax.set_title("各类型 ISR 随噪音变化", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)


def main():
    pairs = [
        ("exp3_case_study_zh_*.json", "zh_main", "exp3_case_types_zh_main.png"),
        ("exp3_case_study_en_*.json", "en_main", "exp3_case_types_en_main.png"),
        ("exp3_case_study_zh_fact_*.json", "zh_fact", "exp3_case_types_zh_fact.png"),
        ("exp3_case_study_en_fact_*.json", "en_fact", "exp3_case_types_en_fact.png"),
    ]

    for pattern, label, fname in pairs:
        files = sorted(RESULTS.glob(pattern))
        if not files:
            print(f"  no files for {pattern}")
            continue
        latest = files[-1]
        data = json.loads(latest.read_text(encoding="utf-8"))
        print(f"  {label}: {latest.name} (n={data.get('n_samples', '?')})")
        make_figure(data, label, FIG / fname)

    # 同时输出到 report_latex/figures
    latex_fig = ROOT / "report_latex" / "figures"
    latex_fig.mkdir(parents=True, exist_ok=True)
    for fname in ["exp3_case_types_zh_main.png", "exp3_case_types_en_main.png",
                   "exp3_case_types_zh_fact.png", "exp3_case_types_en_fact.png"]:
        src = FIG / fname
        if src.exists():
            import shutil
            shutil.copy(src, latex_fig / fname)
            print(f"  copied -> {latex_fig / fname}")

    print("\ndone.")


if __name__ == "__main__":
    main()
