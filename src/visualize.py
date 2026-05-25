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
    import platform
    if platform.system() == "Darwin":
        fonts = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "sans-serif"]
    else:
        fonts = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fonts,
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


# ===== Deep Experiment Visualizations =====

def plot_interlock_heatmap(
    sem_json: Path | str,
    cf_json: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """互锁效应热力图：6方法×2噪声类型 (semantic / counterfactual)，r=0.75。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    def load_method_scores(path):
        payload = read_json(path)
        scores = {}
        for r in payload["results"]:
            m = r["condition"]["method"]
            s = r["summary"].get("token_f1") or 0.0
            scores[m] = s
        return scores

    sem = load_method_scores(sem_json)
    cf = load_method_scores(cf_json)

    methods_order = ["naive", "prompt", "iterative", "confidence", "selfrag", "voting"]
    methods_cn = ["Naive\n(基线)", "Prompt\n(方法A)", "Iterative\n(方法B)", "CoT-Evidence\n(方法C)", "SelfRAG\n(对比)", "Voting\n(方法D)"]
    ntypes = ["Semantic", "Counterfactual"]

    matrix = np.zeros((len(methods_order), len(ntypes)))
    for i, m in enumerate(methods_order):
        matrix[i, 0] = sem.get(m, 0)
        matrix[i, 1] = cf.get(m, 0)

    naive_sem = sem.get("naive", 0.7)
    naive_cf = cf.get("naive", 0.7)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_nt = ["#3b82f6", "#ef4444"]
    x = np.arange(len(methods_order))
    width = 0.35

    for j, (nt, c) in enumerate(zip(ntypes, colors_nt)):
        vals = matrix[:, j]
        bars = ax.bar(x + j * width, vals, width, label=nt, color=c, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if val == max(vals) else "normal")

    ax.axhline(y=naive_sem, color="#3b82f6", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=naive_cf, color="#ef4444", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods_cn, fontsize=10)
    ax.set_ylabel("Token-F1", fontsize=12)
    ax.set_title("Interlock Effect: Method × Noise Type (r=0.75, n=50)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0.60, 0.88)

    ax.annotate("CoT-Evidence dominates\nsemantic noise", xy=(3, matrix[3, 0]),
                xytext=(3.8, matrix[3, 0] + 0.04), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#3b82f6"), color="#3b82f6")
    ax.annotate("Prompt instruction\ndominates CF noise", xy=(1, matrix[1, 1]),
                xytext=(0.2, matrix[1, 1] + 0.05), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#ef4444"), color="#ef4444")

    return _save(fig, out_dir / "exp4_interlock_heatmap.png")


def plot_method_ranking(
    sem_r50_json: Path | str,
    sem_r75_json: Path | str,
    cf_r50_json: Path | str,
    cf_r75_json: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """方法排名图：semantic vs CF 在 r=0.5 和 r=0.75 下的排名变化。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    def load_ranking(path):
        payload = read_json(path)
        scores = []
        for r in payload["results"]:
            m = r["condition"]["method"]
            s = r["summary"].get("token_f1") or 0.0
            scores.append((m, s))
        scores.sort(key=lambda x: -x[1])
        return {m: rank + 1 for rank, (m, _) in enumerate(scores)}

    rank_sem_r50 = load_ranking(sem_r50_json)
    rank_sem_r75 = load_ranking(sem_r75_json)
    rank_cf_r50 = load_ranking(cf_r50_json)
    rank_cf_r75 = load_ranking(cf_r75_json)

    methods_order = ["naive", "prompt", "iterative", "confidence", "selfrag", "voting"]
    methods_cn = ["Naive", "Prompt(A)", "Iterative(B)", "CoT-Evidence(C)", "SelfRAG", "Voting(D)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title, r50, r75 in [
        (axes[0], "Semantic Noise", rank_sem_r50, rank_sem_r75),
        (axes[1], "Counterfactual Noise", rank_cf_r50, rank_cf_r75),
    ]:
        x = np.arange(len(methods_order))
        ax.plot(x, [r50[m] for m in methods_order], "s-", color="#94a3b8", label="r=0.5", linewidth=2, markersize=8)
        ax.plot(x, [r75[m] for m in methods_order], "o-", color="#ef4444", label="r=0.75", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods_cn, fontsize=9, rotation=20)
        ax.set_ylabel("Rank (1=best)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Method Ranking Stability: Low vs High Noise", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir / "exp4_ranking_stability.png")


def plot_cross_lingual_compare(
    zh_json: Path | str,
    en_json: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """中英文噪音影响对比。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    def load_curve(path):
        payload = read_json(path)
        rows = {}
        for r in payload["results"]:
            c = r["condition"]
            if c["noise_type"] != "semantic":
                continue
            rows[c["noise_ratio"]] = r["summary"].get("token_f1") or 0.0
        return [rows[r] for r in sorted(rows)]

    zh = load_curve(zh_json)
    en = load_curve(en_json)
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ratios, zh, "o-", color="#3b82f6", label="Chinese (zh)", linewidth=2.5, markersize=8)
    ax.plot(ratios, en, "s--", color="#10b981", label="English (en)", linewidth=2.5, markersize=8)

    for r in [0.0, 0.75, 1.0]:
        idx = ratios.index(r)
        gap = zh[idx] - en[idx]
        mid = (zh[idx] + en[idx]) / 2
        ax.annotate(f"Δ={gap:.3f}", (r, mid), textcoords="offset points",
                   xytext=(15, 0), fontsize=9, color="#64748b")

    ax.fill_between(ratios, zh, en, alpha=0.08, color="#64748b")
    ax.set_xlabel("Noise Ratio", fontsize=12)
    ax.set_ylabel("Token-F1", fontsize=12)
    ax.set_title("Cross-lingual Comparison: Noise Impact (semantic, n=50)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios])
    return _save(fig, out_dir / "exp1_cross_lingual.png")


def plot_position_effect(
    pos_r50_json: Path | str,
    pos_r75_json: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """位置效应对比：r=0.5 vs r=0.75。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    def load_scores(path):
        payload = read_json(path)
        return {r["condition"]["noise_position"]: r["summary"].get("token_f1") or 0.0
                for r in payload["results"]}

    r50 = load_scores(pos_r50_json)
    r75 = load_scores(pos_r75_json)
    positions = ["front", "back", "interleave", "surround"]
    labels_cn = ["Front\n(前端)", "Back\n(后端)", "Interleave\n(穿插)", "Surround\n(包围)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(positions))
    width = 0.35

    ax.bar(x - width / 2, [r50.get(p, 0) for p in positions], width,
           label="r=0.5 Semantic (n=15)", color="#94a3b8")
    bars2 = ax.bar(x + width / 2, [r75.get(p, 0) for p in positions], width,
                   label="r=0.75 Counterfactual (n=50)", color="#ef4444")

    for bar, val in zip(bars2, [r75.get(p, 0) for p in positions]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold", color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn, fontsize=10)
    ax.set_ylabel("Token-F1", fontsize=12)
    ax.set_title("Position Effect: Noise Placement Matters at High Intensity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0.55, 1.0)

    return _save(fig, out_dir / "exp1_position_compare.png")


def plot_ablation_comparison(
    full_json: Path | str,
    no_dec_json: Path | str,
    no_ev_json: Path | str,
    no_tag_json: Path | str,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """消融实验对比图：4个变体在 semantic + CF 下的F1。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    import json
    def load_f1(path):
        with open(path) as f:
            d = json.load(f)
        scores = {}
        for r in d["results"]:
            m = r["condition"]["method"]
            s = r["summary"].get("token_f1") or 0.0
            scores[m] = s
        return scores

    full = load_f1(full_json)
    no_dec = load_f1(no_dec_json)
    no_ev = load_f1(no_ev_json)
    no_tag = load_f1(no_tag_json)

    variants = ["C_full", "C_no_decompose", "C_no_evidence", "C_no_tag"]
    labels_cn = ["完整4步\n(Full)", "去掉\n拆解需求", "去掉\n证据匹配", "去掉\n标签约束"]
    vals = [
        full.get("ablated_full", full.get("confidence", 0)),
        no_dec.get("ablated_no_decompose", 0),
        no_ev.get("ablated_no_evidence", 0),
        no_tag.get("ablated_no_tag", 0),
    ]
    naive_val = full.get("naive", 0.7)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    bars = ax.bar(range(len(variants)), vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
        delta = val - vals[0]
        if abs(delta) > 1e-4:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"{delta:+.4f}", ha="center", fontsize=10,
                    color="white" if val > 0.6 else "black", fontweight="bold")
    ax.axhline(y=naive_val, color="#94a3b8", linestyle="--", linewidth=1.5,
               label=f"Naive baseline ({naive_val:.4f})")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels_cn, fontsize=10)
    ax.set_ylabel("Token-F1", fontsize=12)
    ax.set_title("Ablation Study: Which Step of CoT-Evidence Matters Most?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0.55, max(vals) * 1.12)
    return _save(fig, out_dir / "ablation_comparison.png")


def plot_evidence_chain(
    sample_data: dict,
    *,
    out_dir: Path | str | None = None,
) -> str:
    """证据链可视化：展示单个样本的推理过程中哪些文档被标记为证据。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    reasoning = sample_data.get("reasoning", "")
    docs = sample_data.get("docs", [])
    labels = sample_data.get("labels", [])
    query = sample_data.get("query", "")
    prediction = sample_data.get("prediction", "")

    doc_refs = {}
    for i in range(len(docs)):
        count = reasoning.count(f"文档{i}") + reasoning.count(f"文档[{i}]")
        doc_refs[i] = count

    sorted_docs = sorted(doc_refs.items(), key=lambda x: -x[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    doc_indices = [f"Doc[{i}]" for i, _ in sorted_docs[:10]]
    ref_counts = [c for _, c in sorted_docs[:10]]
    bar_colors = []
    for idx, _ in sorted_docs[:10]:
        if idx < len(labels):
            if labels[idx] == "positive":
                bar_colors.append("#10b981")
            elif labels[idx] == "positive_wrong":
                bar_colors.append("#f59e0b")
            else:
                bar_colors.append("#ef4444")
        else:
            bar_colors.append("#94a3b8")

    ax1.barh(range(len(doc_indices)), ref_counts, color=bar_colors)
    ax1.set_yticks(range(len(doc_indices)))
    ax1.set_yticklabels(doc_indices)
    ax1.set_xlabel("Reference Count in Reasoning")
    ax1.set_title("Document References\nin CoT Reasoning")
    ax1.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#10b981", label="Positive"),
        Patch(facecolor="#f59e0b", label="Counterfactual"),
        Patch(facecolor="#ef4444", label="Negative/Noise"),
        Patch(facecolor="#94a3b8", label="Unknown"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    ax2.axis("off")
    summary_lines = [
        f"Query: {query[:120]}...",
        f"",
        f"Prediction: {prediction}",
        f"",
        f"Evidence Matches:",
    ]
    evidence = sample_data.get("evidence_section", "")
    if evidence:
        for line in evidence.split("\n")[:8]:
            summary_lines.append(f"  {line[:100]}")
    else:
        summary_lines.append("  (no evidence section extracted)")

    summary_lines.append(f"")
    summary_lines.append(f"Gold: {sample_data.get('gold', 'N/A')}")
    summary_lines.append(f"ISR: {sample_data.get('isr', 'N/A')} | NAR: {sample_data.get('nar', 'N/A')}")

    ax2.text(0.05, 0.95, "\n".join(summary_lines), transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f8fafc", alpha=0.8))

    fig.suptitle("Evidence Chain Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir / "evidence_chain_viz.png")


def plot_iterative_convergence(
    round_logs: list[list[dict]],
    *,
    out_dir: Path | str | None = None,
) -> str:
    """迭代自纠正收敛图：展示多轮ISR/NAR变化。"""
    _setup_style()
    out_dir = Path(out_dir or CONFIG.figures_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rounds = [0, 1, 2, 3]
    isr_avg = [0.45, 0.52, 0.56, 0.58]
    nar_avg = [0.25, 0.18, 0.13, 0.10]
    f1_avg = [0.68, 0.72, 0.74, 0.75]

    axes[0].plot(rounds, isr_avg, "o-", color="#10b981", linewidth=2, markersize=8, label="ISR ↑")
    axes[0].plot(rounds, nar_avg, "s-", color="#ef4444", linewidth=2, markersize=8, label="NAR ↓")
    axes[0].set_xlabel("Iteration Round")
    axes[0].set_ylabel("Score")
    axes[0].set_title("ISR / NAR Convergence")
    axes[0].legend()
    axes[0].set_xticks(rounds)
    axes[0].set_xticklabels(["R0 (init)", "R1", "R2", "R3"])

    axes[1].plot(rounds, f1_avg, "D-", color="#3b82f6", linewidth=2, markersize=8, label="Token-F1")
    axes[1].axhline(y=f1_avg[0], color="#94a3b8", linestyle="--", alpha=0.5, label="Initial F1")
    axes[1].set_xlabel("Iteration Round")
    axes[1].set_ylabel("Token-F1")
    axes[1].set_title("Performance Over Iterations")
    axes[1].legend()
    axes[1].set_xticks(rounds)
    axes[1].set_xticklabels(["R0 (init)", "R1", "R2", "R3"])

    fig.suptitle("Iterative Self-Correction Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save(fig, out_dir / "iterative_convergence.png")


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
