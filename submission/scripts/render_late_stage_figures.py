"""Generate figures for late-stage (cross-dataset) experiments → report_latex/figures/."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.results_paths import glob_results
from src.utils import get_logger, read_json
from src.visualize import (
    _PALETTE,
    _save,
    _setup_style,
    plot_correction_compare,
    plot_method_noise_curves,
    plot_noise_impact,
    plot_nrs_grouped_bar,
    plot_robustness_radar,
    render_batch_run_figures,
)

logger = get_logger("render_late")
REPORT_FIG = ROOT / "report_final" / "figures"
REPORT_FIG.mkdir(parents=True, exist_ok=True)


def _is_canonical_result(path: Path) -> bool:
    """跳过合成 judge 文件，优先真实跑数 JSON。"""
    name = path.name
    if "_judge.json" in name or "_gen.json" in name:
        return False
    try:
        payload = read_json(path)
    except Exception:
        return True
    if payload.get("synthetic"):
        return False
    return True


def _pick(pattern: str, *, n: int | None = None, min_n: int = 0) -> Path | None:
    files = [f for f in sorted(glob_results(pattern)) if _is_canonical_result(f)]
    if not files:
        return None
    if n is None and min_n <= 0:
        return files[-1]
    best: Path | None = None
    best_n = -1
    for f in files:
        try:
            payload = read_json(f)
            fn = int(
                payload.get("n_samples")
                or payload.get("args", {}).get("n")
                or (payload.get("results") or [{}])[0].get("summary", {}).get("n")
                or 0
            )
        except Exception:
            fn = 0
        if n is not None and fn != n:
            continue
        if min_n > 0 and fn < min_n:
            continue
        if fn >= best_n:
            best_n = fn
            best = f
    return best or files[-1]


def _copy(src: Path, dst_name: str) -> str:
    dst = REPORT_FIG / dst_name
    shutil.copy2(src, dst)
    logger.info(f"copy -> {dst}")
    return str(dst)


def plot_method_bar_at_ratio(
    result_json: Path,
    *,
    ratio: float,
    out_path: Path,
    title: str,
    score_key: str | None = None,
) -> str:
    _setup_style()
    data = read_json(result_json)
    if score_key is None:
        score_key = "judge_score" if any(
            (r.get("summary") or {}).get("judge_score") is not None for r in data.get("results", [])
        ) else "token_f1"

    rows: list[tuple[str, float]] = []
    for r in data.get("results", []):
        c = r["condition"]
        if abs(float(c.get("noise_ratio", -1)) - ratio) > 1e-9:
            continue
        s = r["summary"]
        val = s.get(score_key)
        if val is None:
            continue
        rows.append((c["method"], float(val)))
    rows.sort(key=lambda x: -x[1])
    methods = [m for m, _ in rows]
    scores = [s for _, s in rows]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(methods))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, scores, color=colors)
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(score_key.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, max(scores) * 1.2 + 0.05 if scores else 1.0)
    return _save(fig, out_path)


def plot_cross_dataset_judge(
    json_paths: dict[str, Path],
    *,
    out_path: Path,
    method: str = "naive",
) -> str:
    """Cross-dataset naive judge_score vs noise ratio."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, path) in enumerate(json_paths.items()):
        data = read_json(path)
        pts: list[tuple[float, float]] = []
        for r in data.get("results", []):
            c = r["condition"]
            if c.get("method") != method:
                continue
            ratio = c.get("noise_ratio")
            if ratio is None:
                continue
            js = r["summary"].get("judge_score")
            if js is None:
                continue
            pts.append((float(ratio), float(js)))
        pts.sort(key=lambda x: x[0])
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            color=_PALETTE[i % len(_PALETTE)],
            label=label,
            linewidth=2,
        )
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("DeepSeek Judge Score")
    ax.set_title(f"Cross-Dataset Robustness ({method}, semantic, Judge)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_cross_dataset_f1(
    json_paths: dict[str, Path],
    *,
    out_path: Path,
    method: str = "naive",
    noise_type: str = "semantic",
    score_key: str = "token_f1",
) -> str:
    """Cross-dataset naive Token-F1（2Wiki / Cmedqa 主实验真实指标）。"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ylabel = "Token-F1" if score_key == "token_f1" else score_key.replace("_", " ").title()
    for i, (label, path) in enumerate(json_paths.items()):
        data = read_json(path)
        pts: list[tuple[float, float]] = []
        for r in data.get("results", []):
            c = r["condition"]
            if c.get("method") != method or c.get("noise_type") != noise_type:
                continue
            ratio = c.get("noise_ratio")
            val = r["summary"].get(score_key)
            if ratio is None or val is None:
                continue
            pts.append((float(ratio), float(val)))
        pts.sort(key=lambda x: x[0])
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            color=_PALETTE[i % len(_PALETTE)],
            label=label,
            linewidth=2,
        )
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Cross-Dataset Naive Curve ({noise_type}, {ylabel})")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def main() -> None:
    generated: list[tuple[str, str]] = []

    # --- MobileMem smoke n=50 ---
    mobilemem = _pick("smoke_mobilemem_zh_calc_n50*.json")
    if mobilemem:
        out_dir = ROOT / "figures" / "late" / "mobilemem"
        render_batch_run_figures(mobilemem, out_dir=out_dir, tag="mobilemem")
        mapping = {
            "late_mobilemem_noise.png": out_dir / "exp1_noise_impact.png",
            "late_mobilemem_correction.png": out_dir / "exp2_correction.png",
            "late_mobilemem_method_curves.png": out_dir / "method_noise_curves.png",
            "late_mobilemem_radar.png": out_dir / "robustness_radar.png",
        }
        for dst, src in mapping.items():
            if src.exists():
                generated.append((_copy(src, dst), dst))

    # --- 2Wiki ---
    wiki_exp1 = _pick("exp_2wiki_exp1_en_main_*.json", min_n=500)
    wiki_exp2 = _pick("exp_2wiki_exp2_en_main_*.json", min_n=500)
    wiki_exp4 = _pick("exp_2wiki_exp4_en_main_*.json", min_n=500)
    wiki_out = ROOT / "figures" / "late" / "2wiki"
    wiki_out.mkdir(parents=True, exist_ok=True)

    if wiki_exp1:
        p = plot_noise_impact(wiki_exp1, out_dir=wiki_out)
        generated.append((_copy(Path(p), "late_2wiki_noise_impact.png"), "late_2wiki_noise_impact.png"))
    if wiki_exp2:
        render_batch_run_figures(wiki_exp2, out_dir=wiki_out, tag="2wiki_exp2")
        for src, dst in [
            (wiki_out / "exp2_correction.png", "late_2wiki_correction.png"),
            (wiki_out / "method_noise_curves.png", "late_2wiki_method_curves.png"),
            (wiki_out / "robustness_radar.png", "late_2wiki_radar.png"),
        ]:
            if src.exists():
                generated.append((_copy(src, dst), dst))
    if wiki_exp4:
        p = plot_method_bar_at_ratio(
            wiki_exp4,
            ratio=0.75,
            out_path=wiki_out / "exp4_r075.png",
            title="2Wiki: Methods @ semantic r=0.75 (Token-F1, n=500)",
            score_key="token_f1",
        )
        generated.append((_copy(Path(p), "late_2wiki_exp4_r075.png"), "late_2wiki_exp4_r075.png"))

    # --- Cmedqa (Token-F1 pilot n=100; 全量 F1 JSON 丢失) ---
    cmedqa = _pick("exp_new_exp2_cmedqa_zh_main_*.json", min_n=50)
    if cmedqa:
        out_dir = ROOT / "figures" / "late" / "cmedqa"
        render_batch_run_figures(cmedqa, out_dir=out_dir, tag="cmedqa")
        cmedqa_exp1 = _pick("exp_new_exp1_cmedqa_zh_main_*.json", min_n=100)
        if cmedqa_exp1:
            plot_noise_impact(cmedqa_exp1, out_dir=out_dir)
        for src_name, dst in [
            ("exp1_noise_impact.png", "late_cmedqa_noise.png"),
            ("exp2_correction.png", "late_cmedqa_correction.png"),
            ("method_noise_curves.png", "late_cmedqa_method_curves.png"),
        ]:
            src = out_dir / src_name
            if src.exists():
                generated.append((_copy(src, dst), dst))
    ng_rgb = _pick("exp_noise_gradient_rgb_n50_*.json", n=50) or _pick("exp_noise_gradient_rgb_n50_*.json")
    ng_2wiki = _pick("exp_noise_gradient_2wiki_n500_*.json", min_n=500) or _pick("exp_noise_gradient_2wiki_n50_*.json")
    ng_cmedqa = _pick("exp_noise_gradient_cmedqa_n500_*.json", min_n=500) or _pick(
        "exp_noise_gradient_cmedqa_n50_*.json", n=50
    )
    judge_map = {}
    if ng_rgb:
        judge_map["RGB zh"] = ng_rgb
    if ng_2wiki:
        judge_map["2Wiki en"] = ng_2wiki
    if ng_cmedqa:
        judge_map["Cmedqa zh"] = ng_cmedqa
    if judge_map:
        p = plot_cross_dataset_judge(
            judge_map,
            out_path=REPORT_FIG / "late_cross_dataset_judge.png",
        )
        generated.append((p, "late_cross_dataset_judge.png"))

    # --- Cross-dataset F1 (2wiki exp1 + cmedqa exp1 if available) ---
    cmedqa_exp1 = _pick("exp_new_exp1_cmedqa_zh_main_*.json", min_n=50)
    f1_map = {}
    if wiki_exp1:
        f1_map["2Wiki en"] = wiki_exp1
    if cmedqa_exp1:
        f1_map["Cmedqa zh"] = cmedqa_exp1
    if f1_map:
        p = plot_cross_dataset_f1(
            f1_map,
            out_path=REPORT_FIG / "late_cross_dataset_f1.png",
        )
        generated.append((p, "late_cross_dataset_f1.png"))

    print("\n[late-stage figures]")
    for path, name in generated:
        print(f"  {name} <- {path}")


if __name__ == "__main__":
    main()
