"""可视化模块 —— 把实验 JSON 转成论文级图表。

仅依赖 matplotlib + seaborn，CPU 即可。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG
from .utils import get_logger, read_json

logger = get_logger(__name__)

_PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Arial"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _save(fig: plt.Figure, out_path: Path | str) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"figure -> {out_path}")
    return str(out_path)


def plot_noise_impact(result_json: Path | str, *, out_dir: Path | str | None = None) -> str:
    """实验一图：噪音比例-F1 折线（按噪音类型分组）。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)
    payload = read_json(result_json)
    rows: list[dict] = []
    for r in payload["results"]:
        s = r["summary"]
        c = r["condition"]
        rows.append(
            {
                "method": c["method"],
                "ratio": c["noise_ratio"],
                "ntype": c["noise_type"],
                "f1": s.get("token_f1") or 0.0,
                "rouge": s.get("rouge_l") or 0.0,
                "isr": s.get("isr") or 0.0,
                "nar": s.get("nar") or 0.0,
            }
        )

    types = sorted({r["ntype"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(types):
        sub = sorted([r for r in rows if r["ntype"] == t], key=lambda x: x["ratio"])
        ax.plot(
            [r["ratio"] for r in sub],
            [r["f1"] for r in sub],
            "-o",
            color=_PALETTE[i % len(_PALETTE)],
            label=t,
            linewidth=2,
        )
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Token-F1")
    ax.set_title("Exp1: Noise Impact on QA Performance")
    ax.legend(title="Noise Type")
    return _save(fig, out_dir / "exp1_noise_impact.png")


def plot_correction_compare(result_json: Path | str, *, out_dir: Path | str | None = None) -> str:
    """实验二图：5 方法在 4 比例下的 F1 分组柱状图。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)
    payload = read_json(result_json)
    rows: list[dict] = []
    for r in payload["results"]:
        s = r["summary"]
        c = r["condition"]
        rows.append(
            {
                "method": c["method"],
                "ratio": c["noise_ratio"],
                "f1": s.get("token_f1") or 0.0,
            }
        )
    methods = sorted({r["method"] for r in rows})
    ratios = sorted({r["ratio"] for r in rows})
    matrix = np.zeros((len(methods), len(ratios)))
    for r in rows:
        i = methods.index(r["method"])
        j = ratios.index(r["ratio"])
        matrix[i, j] = r["f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ratios))
    width = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            matrix[i],
            width=width,
            label=m,
            color=_PALETTE[i % len(_PALETTE)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Token-F1")
    ax.set_title("Exp2: Correction Methods vs Naive")
    ax.legend(ncol=min(len(methods), 5))
    return _save(fig, out_dir / "exp2_correction.png")


def _aggregate_radar_by_method(robustness_table: list[dict]) -> list[dict]:
    """新版 robustness_table 每个 method 可能有多行（按 noise_type 切片）。

    此函数把同一 method 跨 noise_type 的指标取均值，给雷达图用。
    """
    grouped: dict[str, list[dict]] = {}
    for row in robustness_table:
        grouped.setdefault(row.get("method", "?"), []).append(row)
    out: list[dict] = []
    for method, rows in grouped.items():
        def _avg(key: str) -> float | None:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        out.append(
            {
                "method": method,
                "NS": _avg("NS"),
                "NRS": _avg("NRS"),
                "ISR_avg": _avg("ISR_avg") or 0.0,
                "NAR_avg": _avg("NAR_avg") or 0.0,
                "n_rows": len(rows),
            }
        )
    return out


def plot_robustness_radar(robustness_table: list[dict], *, out_path: Path | str) -> str:
    """雷达图：方法-level NS / NRS / ISR / NAR / CRR。"""
    _setup_style()
    rows = _aggregate_radar_by_method(robustness_table)
    methods = [r["method"] for r in rows]
    metrics = ["1-NS", "1-|NRS|", "ISR", "1-NAR"]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    for i, row in enumerate(rows):
        vals = [
            1 - (row.get("NS") or 0),
            1 - abs(row.get("NRS") or 0),
            row.get("ISR_avg") or 0,
            1 - (row.get("NAR_avg") or 0),
        ]
        vals = [max(0.0, min(1.0, v)) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", color=_PALETTE[i % len(_PALETTE)], label=methods[i])
        ax.fill(angles, vals, alpha=0.1, color=_PALETTE[i % len(_PALETTE)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.2, 1.05))
    ax.set_title("Robustness Radar (averaged across noise types)")
    return _save(fig, out_path)


def plot_nrs_grouped_bar(
    robustness_table: list[dict], *, out_path: Path | str, title: str | None = None
) -> str:
    """按 (method, noise_type) 切片的 NRS 条形图。

    斜率绝对值越小越鲁棒。同一 method 三种噪音类型用相邻颜色组。
    """
    _setup_style()
    methods = sorted({r.get("method", "?") for r in robustness_table})
    ntypes = sorted({r.get("noise_type", "?") for r in robustness_table})
    matrix = np.full((len(methods), len(ntypes)), np.nan)
    for r in robustness_table:
        if r.get("NRS") is None:
            continue
        i = methods.index(r["method"])
        j = ntypes.index(r["noise_type"])
        matrix[i, j] = r["NRS"]

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.6), 5))
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(ntypes))
    for j, t in enumerate(ntypes):
        vals = matrix[:, j]
        ax.bar(
            x + j * width - 0.4 + width / 2,
            np.where(np.isnan(vals), 0.0, vals),
            width=width,
            label=t,
            color=_PALETTE[j % len(_PALETTE)],
        )
    ax.axhline(0, color="#0f172a", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("NRS (noise resistance slope, closer to 0 = more robust)")
    ax.set_title(title or "Noise Resistance Slope by Method × Type")
    ax.legend(title="Noise Type", ncol=min(3, len(ntypes)))
    return _save(fig, out_path)


def plot_case_type_distribution(
    cases: list[dict], *, out_path: Path | str, title: str | None = None
) -> str:
    """exp3 案例类型分布饼图（Type1 矫正生效 / 未生效 / Type2 / Type3 / Type4）。"""
    _setup_style()
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.get("type", "Other")] = counts.get(c.get("type", "Other"), 0) + 1
    order = [
        "Type1-矫正生效",
        "Type1-矫正未生效",
        "Type2-噪音激发",
        "Type4-免疫",
        "Type3-淹没",
        "Other",
    ]
    labels = [t for t in order if t in counts]
    sizes = [counts[t] for t in labels]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p / 100 * sum(sizes)))})",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops=dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("#0f172a")
    ax.set_title(title or "Exp3: Case Type Distribution")
    return _save(fig, out_path)


def plot_isr_nar_scatter(
    rows: list[dict], *, out_path: Path | str, title: str | None = None
) -> str:
    """ISR-NAR 散点图：x=ISR, y=NAR；理想区在右下（高 ISR、低 NAR）。

    rows 形如 [{"method":..., "isr":..., "nar":..., "ratio":...}, ...]。
    """
    _setup_style()
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, m in enumerate(methods):
        sub = [r for r in rows if r["method"] == m]
        ax.scatter(
            [r.get("isr", 0.0) for r in sub],
            [r.get("nar", 0.0) for r in sub],
            color=_PALETTE[i % len(_PALETTE)],
            label=m,
            s=60,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("ISR (Info-Source Rate)")
    ax.set_ylabel("NAR (Noise Adoption Rate)")
    ax.set_title(title or "ISR vs NAR by Method")
    ax.legend(title="Method")
    return _save(fig, out_path)


def render_all_from_results_dir(results_dir: Path | str | None = None) -> list[str]:
    results_dir = Path(results_dir or CONFIG.results_dir)
    out_paths: list[str] = []
    for jf in sorted(results_dir.glob("*.json")):
        name = jf.stem
        if name.startswith("exp1_"):
            out_paths.append(plot_noise_impact(jf))
        elif name.startswith("exp2_"):
            out_paths.append(plot_correction_compare(jf))
        elif name.startswith("exp4_"):
            payload = read_json(jf)
            tbl = payload.get("robustness_table", [])
            if tbl:
                out_paths.append(
                    plot_robustness_radar(tbl, out_path=CONFIG.figures_dir / f"{name}_radar.png")
                )
    return out_paths
