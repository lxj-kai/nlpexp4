"""打印 exp3（案例深度分析·大样本版）汇总 + 出图。"""
from __future__ import annotations

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

TYPE_ORDER = ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫", "Other"]
TYPE_COLORS = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#94a3b8"]

# ── 找最新 exp3 结果 ──
files = sorted((ROOT / "experiments" / "results").glob("exp3_case_study_*.json"))
if not files:
    print("no exp3 results found")
    sys.exit(0)

latest = files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
n_samples = data.get("n_samples", "?")
print(f"file: {latest.name}  (N={n_samples})\n")

# ── 1. 总体汇总 ──
print("=" * 80)
print("EXP3 · 案例深度分析汇总 (全量分类)")
print("=" * 80)

results = data["results"]
clean = results[0]
s = clean["summary"]
print(f"\nclean baseline (naive, r=0.0):")
print(f"  n={s.get('n')} | F1={s.get('token_f1',0):.4f} | contains={s.get('contains',0):.3f} | "
      f"ISR={s.get('isr',0):.4f}")

# ── 按 ratio 汇总 ──
extra = data.get("ratio_summaries", {})
if extra:
    ratios = sorted(float(k) for k in extra.keys())
    print(f"\n{'ratio':>6}  {'N':>5}  {'F1_naive':>9}  {'F1_corr':>9}  ", end="")
    for tn in TYPE_ORDER:
        short = tn.replace("Type1-", "生效").replace("Type3-", "淹没").replace("Type4-", "免疫")
        print(f"{short:>10}", end="  ")
    print()
    print("-" * 110)

    for r in ratios:
        noisy_res = results[1 + ratios.index(r) * 2]
        corr_res = results[1 + ratios.index(r) * 2 + 1]
        f1_n = noisy_res["summary"].get("token_f1", 0) or 0
        f1_c = corr_res["summary"].get("token_f1", 0) or 0
        rs = extra[str(r)]
        n_total = rs.get("n_total", 0)
        props = rs.get("type_proportions", {})
        print(f"  {r:.2f}  {n_total:>5}  {f1_n:>9.4f}  {f1_c:>9.4f}  ", end="")
        for tn in TYPE_ORDER:
            pct = props.get(tn, 0.0)
            count = rs.get("type_distribution", {}).get(tn, 0)
            print(f"{pct:>5.1f}%({count:>3})", end="  ")
        print()

# ── 2. Per-type 指标 ──
print(f"\n{'='*80}")
print("Per-type 指标分解 (最后一个 ratio)")
print(f"{'='*80}")
last_r = ratios[-1] if ratios else None
if last_r and str(last_r) in extra:
    ptype_stats = extra[str(last_r)].get("per_type_stats", {})
    for tn in TYPE_ORDER:
        st = ptype_stats.get(tn, {})
        if not st:
            continue
        print(f"\n  {tn}: n={st.get('count','?')} ({st.get('pct','?')}%)")
        print(f"    avg F1:    clean={st.get('avg_f1_clean','?')} -> noisy={st.get('avg_f1_noisy','?')} -> corrected={st.get('avg_f1_corrected','?')}")
        print(f"    avg ISR:    clean={st.get('avg_isr_clean','?')} -> noisy={st.get('avg_isr_noisy','?')}")
        print(f"    avg NAR:    noisy={st.get('avg_nar_noisy','?')} -> corrected={st.get('avg_nar_corrected','?')}")
        print(f"    contains:   clean={st.get('contains_rate_clean','?')} -> noisy={st.get('contains_rate_noisy','?')} -> corrected={st.get('contains_rate_corrected','?')}")

# ── 3. 跨比例过渡矩阵 ──
trans = data.get("transition_matrix", [])
if trans:
    print(f"\n{'='*80}")
    print("跨比例案例类型过渡")
    print(f"{'='*80}")
    for tm in trans:
        print(f"\n  r={tm['from_ratio']} -> r={tm['to_ratio']}:")
        for t_from, t_to_counts in sorted(tm.get("matrix", {}).items()):
            total = sum(t_to_counts.values())
            items = ", ".join(f"{t}({n}/{total}={n/total*100:.0f}%)"
                            for t, n in sorted(t_to_counts.items(), key=lambda x: -x[1]))
            print(f"    {t_from}: {items}")

# ── 4. 展示案例 ──
display_cases = data.get("display_cases", data.get("cases", []))
print(f"\n{'='*80}")
print(f"展示案例: {len(display_cases)} total (前 10 条)")
print(f"{'='*80}")
for case in display_cases[:10]:
    print(f"\n--- #{case['sample_id']} | r={case['noise_ratio']} | {case['type']} ---")
    print(f"  Q: {case['query'][:80]}")
    print(f"  Gold: {case['gold']}")
    print(f"  Clean:     {case['pred_clean']}")
    print(f"  Noisy:     {case['pred_noisy']}")
    print(f"  Corrected: {case['pred_corrected']}")
    print(f"  F1(clean/noisy/corr)={case.get('f1_clean','?')}/{case.get('f1_noisy','?')}/{case.get('f1_corrected','?')}")
    print(f"  ISR(clean/noisy)={case.get('isr_clean','?')}/{case.get('isr_noisy','?')}  "
          f"NAR(noisy/corr)={case.get('nar_noisy','?')}/{case.get('nar_corrected','?')}")

# ── 5. 出图 ──
_setup_style()
out_dir = CONFIG.figures_dir

if extra:
    ratios = sorted(float(k) for k in extra.keys())

    # 图A: 比例堆叠柱状图
    proportions = np.zeros((len(TYPE_ORDER), len(ratios)))
    n_totals = []
    for j, r in enumerate(ratios):
        rs = extra[str(r)]
        n_totals.append(rs.get("n_total", 0))
        for i, tn in enumerate(TYPE_ORDER):
            proportions[i, j] = rs.get("type_proportions", {}).get(tn, 0.0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(ratios))
    x = np.arange(len(ratios))
    for i, tn in enumerate(TYPE_ORDER):
        bars = ax.bar(x, proportions[i], bottom=bottom,
                      color=TYPE_COLORS[i], label=tn, width=0.55,
                      edgecolor="white", linewidth=0.8)
        for j, (bar, pct) in enumerate(zip(bars, proportions[i])):
            if pct > 6:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[j] + pct / 2,
                        f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
        bottom += proportions[i]
    for j, nt in enumerate(n_totals):
        ax.text(j, 103, f"n={nt}", ha="center", va="bottom", fontsize=8.5, color="#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.0%}" for r in ratios], fontsize=11)
    ax.set_xlabel("噪音比例", fontsize=12)
    ax.set_ylabel("案例占比 (%)", fontsize=12)
    ax.set_title(f"Exp3: 案例类型分布 (全量分类, N={n_totals[0] if n_totals else '?'})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 112)
    ax.axhline(y=100, color="#94a3b8", linestyle="--", alpha=0.4, linewidth=0.8)
    _save(fig, out_dir / "exp3_case_types.png")
    print(f"\nfigure A: case type distribution -> exp3_case_types.png")

    # 图B: 性能折线
    f1_clean_val = s.get("token_f1", 0) or 0
    f1_noisy_vals = []
    f1_corr_vals = []
    for r in ratios:
        idx = ratios.index(r)
        f1_noisy_vals.append((results[1 + idx * 2]["summary"].get("token_f1") or 0))
        f1_corr_vals.append((results[1 + idx * 2 + 1]["summary"].get("token_f1") or 0))

    x_full = range(len(ratios) + 1)
    x_labels_full = ["0%\n(clean)"] + [f"{r:.0%}" for r in ratios]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_full, [f1_clean_val] + f1_noisy_vals, "-o", color=_PALETTE[3],
            label="Naive (无矫正)", linewidth=2)
    ax.plot(x_full, [f1_clean_val] + f1_corr_vals, "-s", color=_PALETTE[1],
            label="Confidence (矫正后)", linewidth=2)
    ax.axhline(y=f1_clean_val, color=_PALETTE[0], linestyle="--", alpha=0.5,
               label="Clean baseline")
    ax.set_xticks(x_full)
    ax.set_xticklabels(x_labels_full)
    ax.set_xlabel("噪音比例")
    ax.set_ylabel("Token-F1")
    ax.set_title(f"Exp3: 噪音影响与矫正恢复 (N={n_samples})", fontsize=13, fontweight="bold")
    ax.legend()
    _save(fig, out_dir / "exp3_performance.png")
    print(f"figure B: performance -> exp3_performance.png")
