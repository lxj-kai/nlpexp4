"""Exp3 案例根因分析：对比 Type1-矫正生效 vs Type1-矫正未生效。

分析为什么 confidence corrector 对某些案例有效而对另一些无效，
提取文档级证据并生成可用于报告的表格。
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"

EXP3_FILES = {
    "zh_main": "exp3_case_study_zh_main_20260525_225028.json",
    "en_main": "exp3_case_study_en_main_20260525_225153.json",
    "zh_fact": "exp3_case_study_zh_fact_20260525_222636.json",
    "en_fact": "exp3_case_study_en_fact_20260525_222723.json",
}


def load_exp3(label: str) -> dict:
    fname = EXP3_FILES.get(label)
    if not fname:
        raise KeyError(label)
    path = RESULTS / fname
    if not path.exists():
        # try glob
        files = sorted(RESULTS.glob(f"exp3_case_study_{label.replace('_', '_')}_*.json"))
        if not files:
            raise FileNotFoundError(f"No exp3 result for {label}")
        path = files[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_cases(data: dict, label: str) -> dict[str, Any]:
    """分析单个 exp3 结果的案例根因。"""
    results = data["results"]
    rs = data["ratio_summaries"]

    # Focus on r=1.0 (max noise where effects are clearest)
    r_key = "1.0"
    if r_key not in rs:
        return {}

    clean_rows = {r["sample_id"]: r for r in results[0]["rows"]}
    # find noisy_r1.0 and corrected_r1.0
    noisy_rows = {}
    corr_rows = {}
    for res in results:
        cond = res["condition"]
        if cond.get("noise_ratio") == 1.0:
            if cond["method"] == "naive":
                noisy_rows = {r["sample_id"]: r for r in res["rows"]}
            else:
                corr_rows = {r["sample_id"]: r for r in res["rows"]}

    # Classify all cases
    type1_eff = []  # Type1-矫正生效
    type1_ineff = []  # Type1-矫正未生效

    for sid, c in clean_rows.items():
        n = noisy_rows.get(sid)
        cor = corr_rows.get(sid)
        if n is None:
            continue
        c_ok = c.get("contains")
        n_ok = n.get("contains")
        cor_ok = cor.get("contains") if cor else False

        if c_ok and not n_ok:
            case_info = {
                "sid": sid,
                "query": c["query"],
                "gold": c["gold"],
                "pred_clean": c["prediction"],
                "pred_noisy": n["prediction"],
                "pred_corrected": cor["prediction"] if cor else None,
                "f1_clean": c.get("token_f1"),
                "f1_noisy": n.get("token_f1"),
                "f1_corrected": cor.get("token_f1") if cor else None,
                "nar_noisy": n.get("nar"),
                "nar_corrected": cor.get("nar") if cor else None,
                "isr_noisy": n.get("isr"),
                "docs_clean": c.get("docs", []),
                "docs_noisy": n.get("docs", []),
            }
            if cor_ok:
                type1_eff.append(case_info)
            else:
                type1_ineff.append(case_info)

    # Analyze differences
    analysis = {
        "label": label,
        "n_type1_eff": len(type1_eff),
        "n_type1_ineff": len(type1_ineff),
        "type1_eff": type1_eff,
        "type1_ineff": type1_ineff,
    }

    # Compare metrics
    if type1_eff:
        analysis["eff_avg_f1_clean"] = round(sum(c["f1_clean"] for c in type1_eff) / len(type1_eff), 4)
        analysis["eff_avg_f1_noisy"] = round(sum(c["f1_noisy"] for c in type1_eff) / len(type1_eff), 4)
        analysis["eff_avg_f1_corr"] = round(sum(c["f1_corrected"] for c in type1_eff if c["f1_corrected"]) / len(type1_eff), 4)
        analysis["eff_avg_nar_noisy"] = round(sum(c["nar_noisy"] for c in type1_eff if c["nar_noisy"]) / len(type1_eff), 4)
        analysis["eff_avg_nar_corr"] = round(sum(c["nar_corrected"] for c in type1_eff if c["nar_corrected"]) / len(type1_eff), 4)

    if type1_ineff:
        analysis["ineff_avg_f1_clean"] = round(sum(c["f1_clean"] for c in type1_ineff) / len(type1_ineff), 4)
        analysis["ineff_avg_f1_noisy"] = round(sum(c["f1_noisy"] for c in type1_ineff) / len(type1_ineff), 4)
        analysis["ineff_avg_f1_corr"] = round(sum(c["f1_corrected"] for c in type1_ineff if c["f1_corrected"]) / len(type1_ineff), 4)
        analysis["ineff_avg_nar_noisy"] = round(sum(c["nar_noisy"] for c in type1_ineff if c["nar_noisy"]) / len(type1_ineff), 4)
        analysis["ineff_avg_nar_corr"] = round(sum(c["nar_corrected"] for c in type1_ineff if c["nar_corrected"]) / len(type1_ineff), 4)

    return analysis


def print_case_details(case: dict, tag: str) -> None:
    """Print a single case with full context."""
    print(f"\n{'─'*80}")
    print(f"  [{tag}] Q#{case['sid']}: {case['query']}")
    print(f"  Gold: {case['gold']}")
    print(f"  Clean pred:      {case['pred_clean']}  (F1={case.get('f1_clean','?')})")
    print(f"  Noisy pred:      {case['pred_noisy']}  (F1={case.get('f1_noisy','?')}, NAR={case.get('nar_noisy','?')})")
    print(f"  Corrected pred:  {case['pred_corrected']}  (F1={case.get('f1_corrected','?')}, NAR={case.get('nar_corrected','?')})")

    # Show clean docs (positive) - truncated
    print(f"  Positive docs ({len(case.get('docs_clean', []))}):")
    for i, doc in enumerate(case.get("docs_clean", [])[:3]):
        print(f"    P{i+1}: {doc[:120]}...")

    # Show noisy docs difference
    clean_docs = set(case.get("docs_clean", []))
    noisy_docs = case.get("docs_noisy", [])
    noise_only = [d for d in noisy_docs if d not in clean_docs]
    print(f"  Noise docs injected ({len(noise_only)}):")
    for i, doc in enumerate(noise_only[:5]):
        print(f"    N{i+1}: {doc[:150]}...")


def main():
    print("=" * 90)
    print("EXP3 案例根因分析：Type1-矫正生效 vs Type1-矫正未生效")
    print("=" * 90)

    for label in ["zh_main", "en_main", "zh_fact", "en_fact"]:
        try:
            data = load_exp3(label)
        except FileNotFoundError as e:
            print(f"\n  SKIP {label}: {e}")
            continue

        analysis = analyze_cases(data, label)
        if not analysis:
            continue

        n_eff = analysis["n_type1_eff"]
        n_ineff = analysis["n_type1_ineff"]

        print(f"\n{'='*90}")
        print(f"  {label} (r=1.0)")
        print(f"  Type1-矫正生效: {n_eff} cases  |  Type1-矫正未生效: {n_ineff} cases")
        print(f"{'='*90}")

        if n_eff > 0 and n_ineff > 0:
            print(f"\n  ┌─ 指标对比 ─────────────────────────────────────")
            print(f"  │                   生效 (n={n_eff:>3})      未生效 (n={n_ineff:>3})")
            print(f"  ├──────────────────────────────────────────────────")
            print(f"  │ avg F1 clean:      {analysis.get('eff_avg_f1_clean', '?'):>10}        {analysis.get('ineff_avg_f1_clean', '?'):>10}")
            print(f"  │ avg F1 noisy:      {analysis.get('eff_avg_f1_noisy', '?'):>10}        {analysis.get('ineff_avg_f1_noisy', '?'):>10}")
            print(f"  │ avg F1 corrected:  {analysis.get('eff_avg_f1_corr', '?'):>10}        {analysis.get('ineff_avg_f1_corr', '?'):>10}")
            print(f"  │ avg NAR noisy:     {analysis.get('eff_avg_nar_noisy', '?'):>10}        {analysis.get('ineff_avg_nar_noisy', '?'):>10}")
            print(f"  │ avg NAR corrected: {analysis.get('eff_avg_nar_corr', '?'):>10}        {analysis.get('ineff_avg_nar_corr', '?'):>10}")
            print(f"  └──────────────────────────────────────────────────")

            if analysis.get('eff_avg_nar_noisy') and analysis.get('ineff_avg_nar_noisy'):
                nar_diff = analysis['ineff_avg_nar_noisy'] - analysis['eff_avg_nar_noisy']
                print(f"\n  >>> 关键发现: 未生效组的 NAR(noisy) 比生效组高 {nar_diff:.4f}")
                print(f"      噪音采纳率更高 = 模型更依赖噪音文档 = 矫正更难生效")

        # Show 3 representative cases from each type
        eff_cases = analysis.get("type1_eff", [])
        ineff_cases = analysis.get("type1_ineff", [])

        if eff_cases:
            print(f"\n  ▸ Type1-矫正生效 典型案例 (前3):")
            for case in eff_cases[:3]:
                print_case_details(case, "生效")

        if ineff_cases:
            print(f"\n  ▸ Type1-矫正未生效 典型案例 (前3):")
            for case in ineff_cases[:3]:
                print_case_details(case, "未生效")

    print(f"\n{'='*90}")
    print("分析完成")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
