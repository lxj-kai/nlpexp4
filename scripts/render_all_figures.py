"""一键生成全部实验图表（按 results/ 中的最新文件）。"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CONFIG
from src.utils import read_json, get_logger
from src.visualize import (
    _PALETTE,
    _save,
    _setup_style,
    plot_case_type_distribution,
    plot_correction_compare,
    plot_isr_nar_scatter,
    plot_noise_impact,
    plot_nrs_grouped_bar,
    plot_robustness_radar,
)

logger = get_logger("render")
RESULTS = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def _latest(pattern: str) -> Path | None:
    files = sorted(RESULTS.glob(pattern))
    return files[-1] if files else None


def plot_position_bar(json_path: Path, *, out_path: Path) -> str:
    """位置子实验柱状图。"""
    _setup_style()
    data = read_json(json_path)
    rows = [
        (
            r["condition"]["noise_position"],
            r["summary"].get("token_f1") or 0,
            r["summary"].get("rouge_l") or 0,
            r["summary"].get("contains") or 0,
        )
        for r in data["results"]
    ]
    rows.sort(key=lambda x: ["front", "back", "interleave", "surround"].index(x[0]))
    labels = [r[0] for r in rows]
    f1 = [r[1] for r in rows]
    rl = [r[2] for r in rows]
    cont = [r[3] for r in rows]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, f1, w, label="Token-F1", color=_PALETTE[0])
    ax.bar(x, rl, w, label="ROUGE-L", color=_PALETTE[1])
    ax.bar(x + w, cont, w, label="contains", color=_PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Noise Position")
    ax.set_ylabel("Score")
    ax.set_title("Exp1-Sub: Noise Position Effect (zh/fact, 50% semantic)")
    ax.legend()
    return _save(fig, out_path)


def plot_method_vs_naive(json_path: Path, *, out_path: Path, title: str) -> str:
    """单 ratio 多方法对比柱状图（用于 exp4 反事实场景）。"""
    _setup_style()
    data = read_json(json_path)
    rows = [
        (r["condition"]["method"], r["summary"].get("token_f1") or 0)
        for r in data["results"]
    ]
    methods = [m for m, _ in rows]
    f1 = [s for _, s in rows]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, f1, color=colors)
    for bar, v in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Token-F1")
    ax.set_title(title)
    ax.set_ylim(0, max(f1) * 1.18 + 0.05)
    return _save(fig, out_path)


def main() -> None:
    targets: list[tuple[str, str, str]] = []

    p = _latest("exp1_noise_impact_zh_main_2*.json")
    if p:
        plot_noise_impact(p, out_dir=FIG)
        targets.append(("exp1_main", str(p.name), "exp1_noise_impact.png"))

    p = _latest("exp1_noise_impact_zh_fact_2*.json")
    if p:
        out = plot_noise_impact(p, out_dir=FIG / "_tmp_fact")
        Path(out).replace(FIG / "exp1_noise_impact_zh_fact.png")
        targets.append(("exp1_fact", str(p.name), "exp1_noise_impact_zh_fact.png"))

    p = _latest("exp1_noise_impact_zh_main_position_*.json") or _latest(
        "exp1_noise_impact_zh_fact_position_*.json"
    )
    if p:
        plot_position_bar(p, out_path=FIG / "exp1_position_effect.png")
        targets.append(("exp1_pos", str(p.name), "exp1_position_effect.png"))

    p = _latest("exp2_correction_zh_2*.json")
    if p:
        plot_correction_compare(p, out_dir=FIG)
        targets.append(("exp2", str(p.name), "exp2_correction.png"))

        data = read_json(p)
        tbl = data.get("robustness_table", [])
        if tbl:
            plot_nrs_grouped_bar(
                tbl,
                out_path=FIG / "exp2_nrs_grouped.png",
                title="Exp2: NRS by Method × Noise Type",
            )
            targets.append(("exp2_nrs", str(p.name), "exp2_nrs_grouped.png"))
            plot_robustness_radar(tbl, out_path=FIG / "exp2_robustness_radar.png")
            targets.append(
                ("exp2_radar", str(p.name), "exp2_robustness_radar.png")
            )

        rows = []
        for r in data["results"]:
            cond = r["condition"]
            for row in r.get("rows", []):
                rows.append(
                    {
                        "method": cond["method"],
                        "ratio": cond["noise_ratio"],
                        "isr": row.get("isr") or 0.0,
                        "nar": row.get("nar") or 0.0,
                    }
                )
        noisy_rows = [r for r in rows if r["ratio"] > 0]
        if noisy_rows:
            plot_isr_nar_scatter(
                noisy_rows,
                out_path=FIG / "exp2_isr_nar_scatter.png",
                title="Exp2: ISR vs NAR (noisy conditions only)",
            )
            targets.append(("exp2_isrnar", str(p.name), "exp2_isr_nar_scatter.png"))

    p = _latest("exp3_case_study_zh_*.json")
    if p:
        data = read_json(p)
        cases = data.get("cases", [])
        if cases:
            plot_case_type_distribution(
                cases, out_path=FIG / "exp3_case_type_pie.png"
            )
            targets.append(("exp3_pie", str(p.name), "exp3_case_type_pie.png"))

    p = _latest("exp4_existing_methods_zh_fact_counterfactual_*.json")
    if p:
        plot_method_vs_naive(
            p,
            out_path=FIG / "exp4_counterfactual_methods.png",
            title="Exp4: Correction Methods on Counterfactual Noise (zh/fact, r=0.75)",
        )
        targets.append(("exp4_cf", str(p.name), "exp4_counterfactual_methods.png"))

    p = _latest("exp4_existing_methods_zh_*main*.json") or _latest("exp4_existing_methods_zh_2*.json")
    if p:
        plot_method_vs_naive(
            p,
            out_path=FIG / "exp4_existing_methods.png",
            title="Exp4: Existing Methods on Semantic Noise (zh/main, r=0.5)",
        )
        targets.append(("exp4_main", str(p.name), "exp4_existing_methods.png"))

        data = read_json(p)
        tbl = data.get("robustness_table", [])
        if tbl:
            plot_robustness_radar(tbl, out_path=FIG / "exp4_robustness_radar.png")
            targets.append(("exp4_radar", str(p.name), "exp4_robustness_radar.png"))

    print("\n[generated figures]")
    for tag, src, dst in targets:
        print(f"  {tag:<12} <- {src}\n               -> {dst}")


if __name__ == "__main__":
    main()
