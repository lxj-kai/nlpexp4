"""跨方法案例类型分析。

从 exp3 中选出各类型的典型案例，在相同的噪音上下文中
运行多种矫正器（confidence/prompt/iterative/selfrag），
分析不同方法对不同案例类型的矫正效果。
"""
from __future__ import annotations

import json, sys, argparse
from pathlib import Path
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.data_loader import load_dataset, RGBRecord
from src.noise_injector import batch_inject
from src.rag_pipeline import RAGPipeline
from src.correctors import get_corrector
from src.evaluator import Evaluator, aggregate
from src.llm_client import LLMClient, get_client
from src.utils import get_logger, set_seed

logger = get_logger("cross_method")


def load_exp3_cases(exp3_file: str, ratio: float = 1.0) -> dict[str, list[dict]]:
    """加载 exp3 结果中各类型的展示案例。"""
    data = json.loads(Path(exp3_file).read_text(encoding="utf-8"))
    rs = data.get("ratio_summaries", {}).get(str(ratio), {})
    display = rs.get("display_cases", [])
    if not display:
        return {}

    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in display:
        by_type[c["type"]].append(c)
    return dict(by_type)


def run_cross_method(
    exp3_file: str,
    ratio: float = 1.0,
    max_per_type: int = 5,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """主函数：对 exp3 案例用多种方法矫正，比较效果。"""
    if methods is None:
        methods = ["confidence", "prompt", "iterative", "selfrag"]

    cases_by_type = load_exp3_cases(exp3_file, ratio)
    if not cases_by_type:
        logger.warning("no cases found")
        return {}

    # 选取每类型最多 max_per_type 个
    selected: dict[str, list[dict]] = {}
    for tn in ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]:
        cases = cases_by_type.get(tn, [])
        if cases:
            selected[tn] = cases[:max_per_type]

    # 确定 language 和 subset
    data = json.loads(Path(exp3_file).read_text(encoding="utf-8"))
    args = data.get("args", {})
    language = args.get("language", "zh")
    subset = args.get("subset", "main")

    # 加载对应数据集的全部记录
    set_seed(CONFIG.seed)
    all_records = load_dataset(language=language, subset=subset, shuffle=True)
    records_by_id: dict[int, RGBRecord] = {r.id: r for r in all_records}

    # 收集所有需要测试的 sid
    all_sids = set()
    for cases in selected.values():
        for c in cases:
            all_sids.add(c["sample_id"])

    target_records = [records_by_id[sid] for sid in all_sids if sid in records_by_id]
    if not target_records:
        logger.warning("no matching records found")
        return {}

    logger.info(f"Testing {len(target_records)} records × {len(methods)} methods at r={ratio}")

    # 注入相同的噪音上下文
    noisy_ctxs = batch_inject(
        target_records,
        noise_ratio=ratio,
        noise_type="semantic" if subset in ("main", "refine") else "counterfactual",
        noise_position="interleave",
    )

    llm = get_client()
    evaluator = Evaluator(use_llm_judge=False, llm=llm)

    # 对每种方法运行矫正
    results: dict[str, list[dict]] = {}
    pipe = RAGPipeline(llm=llm)
    naive_results = pipe.batch_answer(noisy_ctxs, language=language, show_progress=False)
    naive_rows = evaluator.evaluate_batch(naive_results)
    naive_by_sid = {r["sample_id"]: r for r in naive_rows}

    for method in methods:
        logger.info(f"  running {method}...")
        corr = get_corrector(method, llm=llm)
        corrected = corr.batch_correct(noisy_ctxs, language=language, show_progress=False)
        rows = evaluator.evaluate_batch(corrected)
        results[method] = rows

    # 对照 clean baseline
    clean_records = [records_by_id[sid] for sid in all_sids if sid in records_by_id]
    clean_ctxs = batch_inject(clean_records, noise_ratio=0.0)
    clean_results = pipe.batch_answer(clean_ctxs, language=language, show_progress=False)
    clean_rows = evaluator.evaluate_batch(clean_results)
    clean_by_sid = {r["sample_id"]: r for r in clean_rows}

    # 构建对比表
    comparison: dict[str, list[dict]] = {}
    for tn, cases in selected.items():
        type_results = []
        for case in cases:
            sid = case["sample_id"]
            c = clean_by_sid.get(sid, {})
            n = naive_by_sid.get(sid, {})
            row = {
                "sid": sid,
                "query": case["query"],
                "gold": case["gold"],
                "type": tn,
                "clean": {
                    "pred": c.get("prediction", ""),
                    "f1": c.get("token_f1", 0),
                    "contains": c.get("contains", False),
                },
                "naive": {
                    "pred": n.get("prediction", ""),
                    "f1": n.get("token_f1", 0),
                    "contains": n.get("contains", False),
                },
            }
            for method in methods:
                method_rows = {r["sample_id"]: r for r in results[method]}
                mr = method_rows.get(sid, {})
                row[method] = {
                    "pred": mr.get("prediction", ""),
                    "f1": mr.get("token_f1", 0),
                    "contains": mr.get("contains", False),
                    "nar": mr.get("nar", 0),
                }
            type_results.append(row)
        comparison[tn] = type_results

    # 按类型聚合统计
    summary: dict[str, dict] = {}
    for tn, rows in comparison.items():
        method_stats = {}
        for method in methods:
            f1s = [r[method]["f1"] for r in rows if r[method]["f1"] is not None]
            contains = [r[method]["contains"] for r in rows]
            nars = [r[method].get("nar", 0) for r in rows]
            method_stats[method] = {
                "avg_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0,
                "contains_rate": round(sum(contains) / len(contains), 4) if contains else 0,
                "avg_nar": round(sum(nars) / len(nars), 4) if nars else 0,
            }
        naive_f1s = [r["naive"]["f1"] for r in rows]
        naive_contains = [r["naive"]["contains"] for r in rows]
        summary[tn] = {
            "n": len(rows),
            "naive": {
                "avg_f1": round(sum(naive_f1s) / len(naive_f1s), 4) if naive_f1s else 0,
                "contains_rate": round(sum(naive_contains) / len(naive_contains), 4) if naive_contains else 0,
            },
            "methods": method_stats,
        }
        # 找出最佳方法
        best = max(method_stats.items(), key=lambda x: x[1]["avg_f1"])
        summary[tn]["best_method"] = best[0]
        summary[tn]["best_f1"] = best[1]["avg_f1"]

    return {
        "comparison": comparison,
        "summary": summary,
        "n_cases": sum(len(v) for v in selected.values()),
        "methods": methods,
        "ratio": ratio,
    }


def print_report(result: dict) -> None:
    """打印分析报告。"""
    if not result:
        print("No results.")
        return

    summary = result["summary"]
    methods = result["methods"]
    comparison = result["comparison"]

    print(f"\n{'='*90}")
    print(f"跨方法案例类型分析 (r={result['ratio']}, {result['n_cases']} cases)")
    print(f"{'='*90}")

    # 汇总表
    print(f"\n┌─ 各类型最佳矫正方法 ──────────────────────────────────────")
    print(f"│ {'案例类型':<20} {'N':>3}  {'最佳方法':<14} {'最佳F1':>8}  {'Naive F1':>8}")
    print(f"├{'─'*60}")
    for tn in ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]:
        if tn not in summary:
            continue
        s = summary[tn]
        print(f"│ {tn:<20} {s['n']:>3}  {s['best_method']:<14} {s['best_f1']:>8.4f}  {s['naive']['avg_f1']:>8.4f}")
    print(f"└{'─'*60}")

    # 各类型详细方法对比
    for tn in ["Type1-矫正生效", "Type1-矫正未生效", "Type3-信息淹没", "Type4-免疫"]:
        if tn not in summary:
            continue
        s = summary[tn]
        print(f"\n▸ {tn} (n={s['n']})")
        print(f"  {'Method':<14} {'avg F1':>8}  {'contains%':>10}  {'avg NAR':>8}")
        print(f"  {'─'*45}")
        print(f"  {'naive':<14} {s['naive']['avg_f1']:>8.4f}  {s['naive']['contains_rate']:>10.1%}  {'-':>8}")
        for m in methods:
            ms = s["methods"][m]
            marker = " *" if m == s["best_method"] else ""
            print(f"  {m:<14} {ms['avg_f1']:>8.4f}  {ms['contains_rate']:>10.1%}  {ms['avg_nar']:>8.4f}{marker}")
        print(f"  * = 最佳方法")

    # 展示一个案例
    if comparison:
        first_type = list(comparison.keys())[0]
        if comparison[first_type]:
            case = comparison[first_type][0]
            print(f"\n▸ 示例案例 [{first_type}] Q#{case['sid']}: {case['query']}")
            print(f"  Gold: {case['gold']}")
            print(f"  Clean:     {case['clean']['pred'][:80]}  (F1={case['clean']['f1']})")
            print(f"  Naive:     {case['naive']['pred'][:80]}  (F1={case['naive']['f1']})")
            for m in methods:
                mr = case[m]
                print(f"  {m:<10}: {mr['pred'][:80]}  (F1={mr['f1']})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp3-file", default=None,
                   help="exp3 result JSON path (default: latest zh/main)")
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--max-per-type", type=int, default=5)
    p.add_argument("--methods", default="confidence,prompt,iterative,selfrag")
    args = p.parse_args()

    if args.exp3_file is None:
        files = sorted(ROOT.glob("experiments/results/exp3_case_study_zh_main_*.json"))
        if not files:
            print("No exp3 results found")
            return
        args.exp3_file = str(files[-1])

    methods = [m.strip() for m in args.methods.split(",")]
    result = run_cross_method(
        args.exp3_file,
        ratio=args.ratio,
        max_per_type=args.max_per_type,
        methods=methods,
    )
    print_report(result)


if __name__ == "__main__":
    main()
