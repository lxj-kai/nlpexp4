"""生成 exp3 案例深度分析图表（大样本版·修正）。

修复：所有类型都标注百分比（包括 <5% 的小段），简化布局确保清晰可读。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CONFIG
from src.visualize import _setup_style, _save
from src.utils import get_logger

logger = get_logger("render_exp3")

RESULTS = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

TYPE_ORDER = ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫", "Other"]
TYPE_COLORS = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#94a3b8"]
TYPE_CN = ["矫正生效", "矫正未生效", "信息淹没", "免疫", "其他(噪音激发)"]


def _lighten_color(hex_color: str, factor: float = 0.5) -> str:
    """返回更浅/更亮的颜色版本，用于在白底上显示白色文字。"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_figure(data: dict, label: str, out_path: Path) -> str:
    _setup_style()
    extra = data.get("ratio_summaries", {})
    results = data["results"]
    if not extra:
        print(f"  no ratio_summaries in {label}")
        return ""

    ratios = sorted(float(k) for k in extra.keys())

    # ── 数据提取 ──
    proportions = np.zeros((len(TYPE_ORDER), len(ratios)))
    counts = np.zeros((len(TYPE_ORDER), len(ratios)), dtype=int)
    n_total_per_ratio = []
    for j, r in enumerate(ratios):
        rs = extra[str(r)]
        n_total_per_ratio.append(rs.get("n_total", 0))
        for i, tn in enumerate(TYPE_ORDER):
            proportions[i, j] = rs.get("type_proportions", {}).get(tn, 0.0)
            counts[i, j] = int(rs.get("type_distribution", {}).get(tn, 0))

    # 哪些类型至少在一个 ratio 中 >0
    active_types = [(i, tn, cn, cl) for i, (tn, cn, cl) in
                    enumerate(zip(TYPE_ORDER, TYPE_CN, TYPE_COLORS))
                    if proportions[i].sum() > 0.01]

    # 性能数据
    f1_clean = results[0]["summary"].get("token_f1") or 0.0
    f1_noisy = [results[1 + i * 2]["summary"].get("token_f1") or 0.0 for i in range(len(ratios))]
    f1_corr = [results[1 + i * 2 + 1]["summary"].get("token_f1") or 0.0 for i in range(len(ratios))]

    # ── 双面板图 (左右) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                                     gridspec_kw={"width_ratios": [1.3, 1]})

    # ── 左：堆叠比例柱状图 (所有段都标注) ──
    bottom = np.zeros(len(ratios))
    x = np.arange(len(ratios))
    ratio_labels = [f"{r:.0%}" for r in ratios]

    for i, tn, cn, color in active_types:
        row = proportions[i]
        bars = ax1.bar(x, row, bottom=bottom, color=color, label=cn,
                       width=0.55, edgecolor="white", linewidth=1.0)
        # 所有非零段都标注（>0.5% 标在段内，<=0.5% 标在段外）
        for j, (bar, pct) in enumerate(zip(bars, row)):
            if pct <= 0.1:
                continue
            if pct > 4:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bottom[j] + pct / 2,
                         f"{pct:.1f}%", ha="center", va="center",
                         fontsize=7.5, fontweight="bold", color="white")
            elif pct > 0.5:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bottom[j] + pct / 2,
                         f"{pct:.1f}", ha="center", va="center",
                         fontsize=6.5, fontweight="bold", color="white")
        bottom += row

    # 标注总数和实际堆积高度
    for j, nt in enumerate(n_total_per_ratio):
        ax1.text(j, max(103, bottom[j] + 2),
                 f"n={nt}", ha="center", va="bottom", fontsize=8.5, color="#64748b")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ratio_labels, fontsize=11)
    ax1.set_xlabel("噪音比例", fontsize=12)
    ax1.set_ylabel("案例占比 (%)", fontsize=12)
    ax1.set_title(f"案例类型分布 ({label})\n全量分类, N≈{n_total_per_ratio[0]}",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9,
               title="案例类型", title_fontsize=9, ncol=2)
    ax1.set_ylim(0, max(115, bottom.max() * 1.15))
    ax1.axhline(y=100, color="#94a3b8", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── 右：性能折线 + 类型占比表格 ──
    # 上半部分: 性能折线
    ax2a = ax2
    x_full = list(range(len(ratios) + 1))
    x_labels_full = ["0%\n(clean)"] + [f"{r:.0%}" for r in ratios]

    ax2a.plot(x_full, [f1_clean] + f1_noisy, "-o", color="#ef4444", linewidth=2.2,
              markersize=8, label="Naive (无矫正)")
    ax2a.plot(x_full, [f1_clean] + f1_corr, "-s", color="#3b82f6", linewidth=2.2,
              markersize=8, label="Confidence (矫正后)")
    ax2a.axhline(y=f1_clean, color="#94a3b8", linestyle="--", alpha=0.6, linewidth=1.5,
                 label=f"Clean baseline ({f1_clean:.3f})")
    ax2a.set_xticks(x_full)
    ax2a.set_xticklabels(x_labels_full, fontsize=10)
    ax2a.set_ylabel("Token-F1", fontsize=12)
    ax2a.set_title(f"噪音影响与矫正恢复 ({label})", fontsize=13, fontweight="bold")
    ax2a.legend(fontsize=9, loc="lower left")
    y_min = max(0, min(min(f1_noisy), min(f1_corr)) * 0.8)
    y_max = max(f1_clean * 1.1, max(f1_corr) * 1.05)
    ax2a.set_ylim(y_min, y_max)

    fig.suptitle(f"Exp3: 案例深度分析 — {label}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = _save(fig, out_path)
    print(f"  -> {path}")
    return path


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
    import shutil
    for fname in ["exp3_case_types_zh_main.png", "exp3_case_types_en_main.png",
                   "exp3_case_types_zh_fact.png", "exp3_case_types_en_fact.png"]:
        src = FIG / fname
        if src.exists():
            shutil.copy(src, latex_fig / fname)
            print(f"  copied -> {latex_fig / fname}")

    print("\ndone.")


if __name__ == "__main__":
    main()
