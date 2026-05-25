"""生成演示用全部图表 + 关键数据汇总。

用法：python scripts/render_demo_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from src.visualize import (
    _setup_style, _save, _PALETTE,
    plot_noise_impact,
    plot_correction_compare,
    plot_interlock_heatmap,
    plot_method_ranking,
    plot_cross_lingual_compare,
    plot_position_effect,
    plot_robustness_radar,
)
from src.config import CONFIG

CONFIG.ensure_dirs()
out_dir = CONFIG.figures_dir
res_dir = ROOT / "experiments" / "results"

print("=" * 60)
print("Generating DEMO figures...")
print("=" * 60)


# ── Reference file paths ──
# We need to pick the LATEST of each experiment type

def latest(pattern: str) -> Path:
    files = sorted(res_dir.glob(pattern))
    if not files:
        print(f"WARNING: no files match {pattern}")
        return None
    return files[-1]


# ── Fig 1: Noise Impact (Chinese, semantic) ──
f1 = latest("exp1_noise_impact_zh_main_20260517*.json")
if f1:
    path = plot_noise_impact(f1)
    print(f"[1/8] Noise Impact (zh) → {path}")

# ── Fig 2: Cross-lingual Comparison ──
f2_zh = latest("exp1_noise_impact_zh_main_20260517*.json")
f2_en = latest("exp1_noise_impact_en_main_*.json")
if f2_zh and f2_en:
    path = plot_cross_lingual_compare(str(f2_zh), str(f2_en))
    print(f"[2/8] Cross-lingual → {path}")

# ── Fig 3: Correction Methods (Chinese, all methods) ──
f3 = latest("exp2_correction_zh_20260518*.json")
if f3:
    path = plot_correction_compare(str(f3))
    print(f"[3/8] Correction Methods (zh) → {path}")

# ── Fig 4: Interlock Heatmap ──
f4_sem = latest("exp4_*semantic_20260525*.json")
f4_cf = latest("exp4_*counterfactual_20260525*.json")
if f4_sem and f4_cf:
    path = plot_interlock_heatmap(str(f4_sem), str(f4_cf))
    print(f"[4/8] Interlock Heatmap → {path}")

# ── Fig 5: Method Ranking Stability ──
f5_sem_r50 = latest("exp4_*semantic_20260524*.json")
f5_sem_r75 = latest("exp4_*semantic_20260525*.json")
f5_cf_r50 = latest("exp4_*counterfactual_20260524*.json")
f5_cf_r75 = latest("exp4_*counterfactual_20260525*.json")
if f5_sem_r50 and f5_sem_r75 and f5_cf_r50 and f5_cf_r75:
    path = plot_method_ranking(str(f5_sem_r50), str(f5_sem_r75), str(f5_cf_r50), str(f5_cf_r75))
    print(f"[5/8] Ranking Stability → {path}")

# ── Fig 6: Position Effect ──
f6_r50 = latest("exp1_*position*20260516*.json")
f6_r75 = latest("exp1_*position*counterfactual_r0.75*.json")
if f6_r50 and f6_r75:
    path = plot_position_effect(str(f6_r50), str(f6_r75))
    print(f"[6/8] Position Effect → {path}")

# ── Fig 7: Correction Methods (English) ──
f7 = latest("exp2_correction_en_*.json")
if f7:
    path = plot_correction_compare(str(f7))
    # rename to distinguish
    import shutil
    dest = out_dir / "exp2_correction_en.png"
    shutil.copy(path, dest)
    print(f"[7/8] Correction Methods (en) → {dest}")

# ── Fig 8: Radar (exp4 semantic r=0.75) ──
if f4_sem:
    payload = json.loads(f4_sem.read_text())
    tbl = payload.get("robustness_table", [])
    if tbl:
        path = plot_robustness_radar(tbl, out_path=out_dir / "exp4_robustness_radar.png")
        print(f"[8/8] Robustness Radar → {path}")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED → figures/")
print("=" * 60)

# ── Print Key Numbers (for presentation slides) ──
print("\n")
print("=" * 60)
print("KEY NUMBERS FOR PRESENTATION")
print("=" * 60)

if f4_sem and f4_cf:
    sem = json.loads(f4_sem.read_text())
    cf = json.loads(f4_cf.read_text())
    sem_scores = {r["condition"]["method"]: r["summary"].get("token_f1", 0) for r in sem["results"]}
    cf_scores = {r["condition"]["method"]: r["summary"].get("token_f1", 0) for r in cf["results"]}

    print(f"\n--- Semantic r=0.75 (n=50) ---")
    for m in sorted(sem_scores, key=lambda x: -sem_scores[x]):
        naive = sem_scores.get("naive", 0.75)
        delta = sem_scores[m] - naive
        print(f"  {m:>12}: {sem_scores[m]:.4f} (Δ naive = {delta:+.4f})")

    print(f"\n--- Counterfactual r=0.75 (n=50) ---")
    for m in sorted(cf_scores, key=lambda x: -cf_scores[x]):
        naive = cf_scores.get("naive", 0.75)
        delta = cf_scores[m] - naive
        print(f"  {m:>12}: {cf_scores[m]:.4f} (Δ naive = {delta:+.4f})")

    print(f"\n--- Interlock Gap ---")
    print(f"  CoT-Evidence: sem={sem_scores.get('confidence',0):.3f} vs CF={cf_scores.get('confidence',0):.3f} "
          f"→ drop {sem_scores.get('confidence',0)-cf_scores.get('confidence',0):.3f}")
    print(f"  Prompt:       sem={sem_scores.get('prompt',0):.3f} vs CF={cf_scores.get('prompt',0):.3f} "
          f"→ gain {cf_scores.get('prompt',0)-sem_scores.get('prompt',0):.3f}")
    print(f"  Voting:       sem={sem_scores.get('voting',0):.3f} vs CF={cf_scores.get('voting',0):.3f} "
          f"→ rank #2 in both")

if f6_r75:
    pos = json.loads(f6_r75.read_text())
    pos_scores = {r["condition"]["noise_position"]: r["summary"].get("token_f1", 0) for r in pos["results"]}
    print(f"\n--- Position Effect (r=0.75 CF) ---")
    for p in ["interleave", "surround", "front", "back"]:
        print(f"  {p:>12}: {pos_scores.get(p, 0):.4f}")
    print(f"  back vs interleave gap: {pos_scores.get('interleave',0) - pos_scores.get('back',0):.3f}")

if f2_zh and f2_en:
    zh_data = json.loads(f2_zh.read_text())
    en_data = json.loads(f2_en.read_text())
    zh_clean = next(r["summary"].get("token_f1", 0) for r in zh_data["results"] if r["condition"]["noise_ratio"] == 0.0)
    en_clean = next(r["summary"].get("token_f1", 0) for r in en_data["results"] if r["condition"]["noise_ratio"] == 0.0)
    print(f"\n--- Cross-lingual Baseline ---")
    print(f"  zh clean F1: {zh_clean:.3f}")
    print(f"  en clean F1: {en_clean:.3f}")
    print(f"  gap: {zh_clean - en_clean:.3f}")

print("\nDone. All figures in figures/ directory.")
