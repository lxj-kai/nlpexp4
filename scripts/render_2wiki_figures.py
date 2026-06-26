"""Render 2WikiMultihopQA (n=500) figures highlighting method differences.

用法:
    python scripts/render_2wiki_figures.py
    python scripts/render_2wiki_figures.py --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results_paths import glob_results  # noqa: E402
from src.utils import get_logger, read_json  # noqa: E402
from src.visualize import (  # noqa: E402
    _PALETTE,
    _save,
    _setup_style,
    plot_correction_compare,
    plot_isr_nar_scatter,
    plot_noise_impact,
)

logger = get_logger("render_2wiki")
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "figures" / "2wiki"

METHOD_ORDER = ["naive", "prompt", "iterative", "confidence", "selfrag", "voting"]
METHOD_LABEL = {
    "naive": "Naive",
    "prompt": "Prompt",
    "iterative": "Iterative",
    "confidence": "CoT-Evidence",
    "selfrag": "Self-RAG",
    "voting": "Voting",
}


def _latest(pattern: str, *, n: int) -> Path | None:
    picked: list[Path] = []
    for f in sorted(glob_results(pattern)):
        data = json.loads(f.read_text(encoding="utf-8"))
        if int(data.get("args", {}).get("n") or 0) == n:
            picked.append(f)
    return picked[-1] if picked else None


def _rows_at_ratio(payload: dict, *, ratio: float, noise_type: str = "semantic") -> list[dict]:
    out: list[dict] = []
    for r in payload["results"]:
        c, s = r["condition"], r["summary"]
        if c.get("noise_type") != noise_type:
            continue
        if float(c.get("noise_ratio", -1)) != ratio:
            continue
        out.append(
            {
                "method": c["method"],
                "ratio": ratio,
                "f1": float(s.get("token_f1") or 0),
                "isr": float(s.get("isr") or 0),
                "nar": float(s.get("nar") or 0),
                "contains": float(s.get("contains") or 0),
            }
        )
    return out


def _ordered(rows: list[dict]) -> list[dict]:
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda r: rank.get(r["method"], 99))


def plot_f1_nar_grouped(rows: list[dict], *, out_path: Path, title: str) -> str:
    """并排柱状：F1 与 NAR，突出 accuracy vs 噪音采纳权衡。"""
    _setup_style()
    rows = _ordered([r for r in rows if r["method"] in METHOD_ORDER])
    methods = [METHOD_LABEL[r["method"]] for r in rows]
    f1 = [r["f1"] for r in rows]
    nar = [r["nar"] for r in rows]

    x = np.arange(len(methods))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, f1, width, color="#3b82f6", alpha=0.88, label="Token-F1")
    bars2 = ax2.bar(x + width / 2, nar, width, color="#ef4444", alpha=0.78, label="NAR")

    naive_f1 = next((r["f1"] for r in rows if r["method"] == "naive"), None)
    naive_nar = next((r["nar"] for r in rows if r["method"] == "naive"), None)
    if naive_f1 is not None:
        ax1.axhline(naive_f1, color="#3b82f6", ls="--", lw=1, alpha=0.5)
    if naive_nar is not None:
        ax2.axhline(naive_nar, color="#ef4444", ls="--", lw=1, alpha=0.5)

    for bar, val in zip(bars1, f1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1d4ed8",
        )
    for bar, val in zip(bars2, nar):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#b91c1c",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("Token-F1 (↑ better)", color="#1d4ed8")
    ax2.set_ylabel("NAR (↓ better)", color="#b91c1c")
    ax1.set_ylim(0, max(f1) * 1.18 if f1 else 1)
    ax2.set_ylim(0, max(max(nar) * 1.25, 0.05) if nar else 0.3)
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_delta_vs_naive(rows: list[dict], *, out_path: Path) -> str:
    """相对 naive 的 ΔF1 / ΔNAR 水平条形图。"""
    _setup_style()
    naive = next((r for r in rows if r["method"] == "naive"), None)
    if not naive:
        raise ValueError("naive baseline missing")

    deltas: list[dict] = []
    for r in _ordered(rows):
        if r["method"] == "naive":
            continue
        deltas.append(
            {
                "method": METHOD_LABEL[r["method"]],
                "df1": r["f1"] - naive["f1"],
                "dnar": r["nar"] - naive["nar"],
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = [d["method"] for d in deltas]
    y = np.arange(len(methods))

    ax = axes[0]
    vals = [d["df1"] for d in deltas]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="#64748b", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Δ Token-F1 vs Naive")
    ax.set_title("Answer Quality Gain", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)

    ax = axes[1]
    vals = [d["dnar"] for d in deltas]
    colors = ["#10b981" if v <= 0 else "#ef4444" for v in vals]
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="#64748b", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Δ NAR vs Naive (negative = less noise adoption)")
    ax.set_title("Noise Suppression", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + (-0.008 if v <= 0 else 0.008), i, f"{v:+.3f}", va="center",
                ha="right" if v <= 0 else "left", fontsize=9)

    fig.suptitle("Method Effect vs Naive Baseline (2Wiki, r=0.75, n=500)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_f1_curves(payload: dict, *, out_path: Path) -> str:
    """各方法 F1 随噪音比例变化折线。"""
    _setup_style()
    series: dict[str, list[tuple[float, float]]] = {}
    for r in payload["results"]:
        c, s = r["condition"], r["summary"]
        if c.get("noise_type") != "semantic":
            continue
        m = c["method"]
        if m not in METHOD_ORDER:
            continue
        series.setdefault(m, []).append((float(c["noise_ratio"]), float(s.get("token_f1") or 0)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, m in enumerate(METHOD_ORDER):
        if m not in series:
            continue
        pts = sorted(series[m])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            color=_PALETTE[i % len(_PALETTE)],
            label=METHOD_LABEL[m],
            linewidth=2.2 if m in ("confidence", "iterative") else 1.8,
            markersize=6,
        )

    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Token-F1")
    ax.set_title("F1 Degradation Curves by Corrector (2Wiki, n=500)", fontweight="bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_pareto_f1_nar(rows: list[dict], *, out_path: Path) -> str:
    """F1–NAR 平面：右上角=高F1低NAR（理想）。"""
    _setup_style()
    rows = _ordered([r for r in rows if r["method"] in METHOD_ORDER])
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for i, r in enumerate(rows):
        m = r["method"]
        ax.scatter(
            r["nar"],
            r["f1"],
            s=140 if m in ("confidence", "iterative") else 100,
            color=_PALETTE[METHOD_ORDER.index(m) % len(_PALETTE)],
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        ax.annotate(
            METHOD_LABEL[m],
            (r["nar"], r["f1"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
            fontweight="bold" if m in ("confidence", "iterative", "voting") else "normal",
        )

    ax.axhline(rows[0]["f1"] if rows else 0.4, color="#94a3b8", ls=":", alpha=0.6)
    ax.axvline(next((r["nar"] for r in rows if r["method"] == "naive"), 0.18), color="#94a3b8", ls=":", alpha=0.6)
    ax.set_xlabel("NAR ↓ better →")
    ax.set_ylabel("Token-F1 ↑ better")
    ax.set_title("F1 vs Noise Adoption Trade-off (r=0.75)", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "Ideal: top-left\n(high F1, low NAR)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f0fdf4", alpha=0.8),
    )
    fig.tight_layout()
    return _save(fig, out_path)


def plot_metric_heatmap(rows: list[dict], *, out_path: Path) -> str:
    """方法 × 指标热力图（r=0.75）。"""
    _setup_style()
    rows = _ordered([r for r in rows if r["method"] in METHOD_ORDER])
    metrics = ["f1", "isr", "contains", "nar"]
    metric_labels = ["Token-F1", "ISR", "Contains", "NAR"]
    matrix = np.array([[r[m] for m in metrics] for r in rows])

    # 按列归一化到 0-1 便于对比；NAR 列反向（低更好）
    norm = matrix.copy()
    for j, m in enumerate(metrics):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-9:
            norm[:, j] = 0.5
        else:
            norm[:, j] = (col - lo) / (hi - lo)
        if m == "nar":
            norm[:, j] = 1 - norm[:, j]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})

    im = axes[0].imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([METHOD_LABEL[r["method"]] for r in rows])
    axes[0].set_title("Relative Strength (green=better)", fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(metrics)):
            axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)

    # 双指标对比：F1 与 (1-NAR) 并排
    x = np.arange(len(rows))
    w = 0.35
    axes[1].bar(x - w / 2, [r["f1"] for r in rows], w, label="F1", color="#3b82f6", alpha=0.85)
    axes[1].bar(x + w / 2, [1 - r["nar"] for r in rows], w, label="1−NAR", color="#10b981", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABEL[r["method"]] for r in rows], rotation=15, ha="right")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Absolute F1 vs Noise Resistance", fontweight="bold")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)

    fig.suptitle("2Wiki Method Profile @ noise ratio 0.75 (n=500)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_summary_panel(exp1: Path, exp4: Path, rows_r75: list[dict], *, out_path: Path) -> str:
    """四宫格总览图。"""
    _setup_style()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # A: exp1 noise curve
    ax = fig.add_subplot(gs[0, 0])
    payload = read_json(exp1)
    curve = sorted(
        [
            (r["condition"]["noise_ratio"], r["summary"].get("token_f1") or 0)
            for r in payload["results"]
            if r["condition"]["method"] == "naive"
        ],
        key=lambda x: x[0],
    )
    ax.plot([c[0] for c in curve], [c[1] for c in curve], "-o", color="#3b82f6", lw=2)
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Token-F1")
    ax.set_title("(A) Naive: Noise Sensitivity", fontweight="bold")

    # B: F1 bars @0.75
    ax = fig.add_subplot(gs[0, 1])
    rows = _ordered([r for r in rows_r75 if r["method"] in METHOD_ORDER])
    ax.bar(
        [METHOD_LABEL[r["method"]] for r in rows],
        [r["f1"] for r in rows],
        color=[_PALETTE[METHOD_ORDER.index(r["method"]) % len(_PALETTE)] for r in rows],
        alpha=0.88,
    )
    ax.axhline(next(r["f1"] for r in rows if r["method"] == "naive"), ls="--", color="#64748b")
    ax.set_ylabel("Token-F1")
    ax.set_title("(B) F1 @ r=0.75", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    # C: NAR bars @0.75
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(
        [METHOD_LABEL[r["method"]] for r in rows],
        [r["nar"] for r in rows],
        color="#ef4444",
        alpha=0.78,
    )
    ax.axhline(next(r["nar"] for r in rows if r["method"] == "naive"), ls="--", color="#64748b")
    ax.set_ylabel("NAR")
    ax.set_title("(C) Noise Adoption @ r=0.75", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    # D: pareto
    ax = fig.add_subplot(gs[1, 1])
    for r in rows:
        m = r["method"]
        ax.scatter(r["nar"], r["f1"], s=90, color=_PALETTE[METHOD_ORDER.index(m) % len(_PALETTE)])
        ax.annotate(METHOD_LABEL[m], (r["nar"], r["f1"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("NAR")
    ax.set_ylabel("Token-F1")
    ax.set_title("(D) F1–NAR Trade-off", fontweight="bold")

    fig.suptitle("2WikiMultihopQA Method Differentiation (n=500, semantic)", fontsize=14, fontweight="bold")
    return _save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()

    exp1 = _latest("exp_2wiki_exp1_en_main_*.json", n=args.n)
    exp2 = _latest("exp_2wiki_exp2_en_main_*.json", n=args.n)
    exp4 = _latest("exp_2wiki_exp4_en_main_*.json", n=args.n)
    if not exp1 or not exp2 or not exp4:
        raise SystemExit(f"missing n={args.n} result JSON under {RESULTS}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exp4_data = read_json(exp4)
    rows_r75 = _rows_at_ratio(exp4_data, ratio=0.75)

    paths: list[str] = []
    paths.append(plot_noise_impact(exp1, out_dir=args.out_dir))
    paths.append(plot_correction_compare(exp2, out_dir=args.out_dir))
    paths.append(
        plot_f1_nar_grouped(
            rows_r75,
            out_path=args.out_dir / "method_f1_nar_r075.png",
            title="Method Comparison @ noise ratio 0.75 (2Wiki, n=500)",
        )
    )
    paths.append(plot_delta_vs_naive(rows_r75, out_path=args.out_dir / "method_delta_vs_naive.png"))
    paths.append(plot_f1_curves(read_json(exp2), out_path=args.out_dir / "method_f1_curves.png"))
    paths.append(plot_pareto_f1_nar(rows_r75, out_path=args.out_dir / "method_pareto_f1_nar.png"))
    paths.append(plot_metric_heatmap(rows_r75, out_path=args.out_dir / "method_metric_heatmap.png"))
    paths.append(
        plot_isr_nar_scatter(
            rows_r75,
            out_path=args.out_dir / "method_isr_nar_scatter.png",
            title="ISR vs NAR @ r=0.75 (ideal: high ISR, low NAR)",
        )
    )
    paths.append(plot_summary_panel(exp1, exp4, rows_r75, out_path=args.out_dir / "method_summary_panel.png"))

    print("figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
