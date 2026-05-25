"""打印 exp3（案例深度分析）汇总 + 出图。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualize import _PALETTE, _setup_style, _save
from src.config import CONFIG

# ── 找最新 exp3 结果 ──
files = sorted((ROOT / "experiments" / "results").glob("exp3_case_study_*.json"))
if not files:
    print("no exp3 results found")
    sys.exit(0)

latest = files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
print(f"file: {latest.name}\n")

# ── 1. 总体汇总 ──
print("=" * 60)
print("EXP3 · 案例深度分析汇总")
print("=" * 60)

results = data["results"]
# clean
clean = results[0]
print(f"\nclean baseline (naive, r=0.0):")
s = clean["summary"]
print(f"  n={s.get('n')} | F1={s.get('token_f1',0):.4f} | contains={s.get('contains',0):.3f} | ISR={s.get('isr',0):.4f}")

# 按 ratio 分组
extra = data.get("ratio_summaries", {})
if extra:
    print(f"\n{'ratio':>6}  {'F1_naive':>10}  {'F1_corr':>10}  {'Type1生效':>10}  {'Type1未生效':>12}  {'Type3':>8}  {'Type4':>8}")
    print("-" * 70)
    for i, r in enumerate(sorted(float(k) for k in extra.keys())):
        noisy_res = results[1 + i * 2]
        corr_res = results[1 + i * 2 + 1]
        f1_n = noisy_res["summary"].get("token_f1", 0) or 0
        f1_c = corr_res["summary"].get("token_f1", 0) or 0
        dist = extra[str(r)]["type_distribution"]
        print(f"  {r:.2f}  {f1_n:>10.4f}  {f1_c:>10.4f}  "
              f"{dist.get('Type1-矫正生效',0):>10}  {dist.get('Type1-矫正未生效',0):>12}  "
              f"{dist.get('Type3-信息淹没',0):>8}  {dist.get('Type4-免疫',0):>8}")

# ── 2. 案例展示 ──
cases = data.get("cases", [])
print(f"\n{'='*60}")
print(f"cases: {len(cases)} total")
print(f"{'='*60}")
for case in cases[:10]:
    print(f"\n--- #{case['sample_id']} | r={case['noise_ratio']} | {case['type']} ---")
    print(f"  Q: {case['query'][:80]}")
    print(f"  Gold: {case['gold']}")
    print(f"  Clean:     {case['pred_clean']}")
    print(f"  Noisy:     {case['pred_noisy']}")
    print(f"  Corrected: {case['pred_corrected']}")
    print(f"  ISR(clean/noisy)={case.get('isr_clean','?')}/{case.get('isr_noisy','?')}  "
          f"NAR(noisy/corr)={case.get('nar_noisy','?')}/{case.get('nar_corrected','?')}")

# ── 3. 出图 ──
_setup_style()
out_dir = CONFIG.figures_dir

# 图A: 各 ratio 下案例类型分布（堆叠柱状图）
if extra:
    ratios = sorted(float(k) for k in extra.keys())
    type_names = ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]
    colors = ["#10b981", "#f59e0b", "#ef4444", "#3b82f6"]

    matrix = np.zeros((len(type_names), len(ratios)))
    for j, r in enumerate(ratios):
        dist = extra[str(r)]["type_distribution"]
        for i, tn in enumerate(type_names):
            matrix[i, j] = dist.get(tn, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(ratios))
    for i, tn in enumerate(type_names):
        ax.bar(range(len(ratios)), matrix[i], bottom=bottom,
               color=colors[i], label=tn, width=0.5)
        bottom += matrix[i]
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.0%}" for r in ratios])
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Exp3: Case Type Distribution by Noise Ratio")
    ax.legend(loc="upper right", fontsize=8)
    out1 = _save(fig, out_dir / "exp3_case_types.png")
    print(f"\nfigure A: {out1}")

# 图B: 各 ratio 下 F1 变化（clean vs noisy vs corrected）
ratios = sorted(float(k) for k in extra.keys())
f1_clean_vals = [s.get("token_f1", 0) or 0]
f1_noisy_vals = []
f1_corr_vals = []
ratio_labels = ["0% (clean)"]
for i, r in enumerate(ratios):
    f1_noisy_vals.append((results[1 + i * 2]["summary"].get("token_f1") or 0))
    f1_corr_vals.append((results[1 + i * 2 + 1]["summary"].get("token_f1") or 0))
    ratio_labels.append(f"{r:.0%}")

x = range(len(ratio_labels))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, [f1_clean_vals[0]] + f1_noisy_vals, "-o", color=_PALETTE[3], label="naive (noisy)", linewidth=2)
ax.plot(x, [f1_clean_vals[0]] + f1_corr_vals, "-s", color=_PALETTE[1], label="confidence (corrected)", linewidth=2)
ax.axhline(y=f1_clean_vals[0], color=_PALETTE[0], linestyle="--", alpha=0.5, label="clean baseline")
ax.set_xticks(x)
ax.set_xticklabels(ratio_labels)
ax.set_xlabel("Noise Ratio")
ax.set_ylabel("Token-F1")
ax.set_title("Exp3: Performance Degradation & Recovery Across Noise Levels")
ax.legend()
out2 = _save(fig, out_dir / "exp3_performance.png")
print(f"figure B: {out2}")
