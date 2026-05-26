"""实验三：案例深度分析（大样本版）。

针对全部/大批样本，同时跑 clean / noisy / corrected，
在多个噪音比例 (0.5-1.0) 下循环，分类所有案例并输出统计。

核心变更（vs 旧版）：
- 默认使用全部数据（--n 0 = 不限制）
- 分类所有案例，不再只 pick top-K，type 分布基于全量统计
- 增加 per-type 指标分解（F1/contains/ISR/NAR）
- 增加 cross-ratio 过渡矩阵（案例在不同比例下的类型变化）

用法：
    python -m experiments.exp3_case_study  # 默认 n=全部, 全量分类
    python -m experiments.exp3_case_study --n 200 --language en --subset main
    python -m experiments.exp3_case_study --n 100 --language zh --subset fact --ratios 0.25,0.5,0.75
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.utils import get_logger

logger = get_logger("exp3")

TYPE_ORDER = ["Type1-矫正生效", "Type1-矫正未生效", "Type2-噪音激发", "Type3-信息淹没", "Type4-免疫", "Other"]


def _classify_case(c: dict, n: dict, cor: dict | None) -> str:
    """对单条案例分类。"""
    if c.get("contains") and not n.get("contains"):
        if cor and cor.get("contains"):
            return "Type1-矫正生效"
        return "Type1-矫正未生效"
    elif c.get("contains") and n.get("contains"):
        return "Type4-免疫"
    elif not c.get("contains") and n.get("contains"):
        return "Type2-噪音激发"
    elif not c.get("contains") and not n.get("contains"):
        return "Type3-信息淹没"
    return "Other"


def _classify_all_cases(
    naive_clean: dict, naive_noisy: dict, corrected: dict, *, ratio: float
) -> list[dict]:
    """分类所有匹配到的样本（不做数量截断）。"""
    cases: list[dict] = []
    for sid, c in naive_clean.items():
        n = naive_noisy.get(sid)
        cor = corrected.get(sid)
        if n is None:
            continue
        type_label = _classify_case(c, n, cor)
        cases.append(
            {
                "sample_id": sid,
                "noise_ratio": ratio,
                "type": type_label,
                "query": c.get("query", ""),
                "gold": c.get("gold"),
                "pred_clean": c.get("prediction"),
                "pred_noisy": n.get("prediction"),
                "pred_corrected": cor.get("prediction") if cor else None,
                "isr_clean": c.get("isr"),
                "isr_noisy": n.get("isr"),
                "nar_noisy": n.get("nar"),
                "nar_corrected": cor.get("nar") if cor else None,
                "f1_clean": c.get("token_f1"),
                "f1_noisy": n.get("token_f1"),
                "f1_corrected": cor.get("token_f1") if cor else None,
                "contains_clean": c.get("contains"),
                "contains_noisy": n.get("contains"),
                "contains_corrected": cor.get("contains") if cor else None,
            }
        )
    return cases


def _compute_per_type_stats(cases: list[dict]) -> dict[str, dict]:
    """按类型聚合指标统计。

    Returns:
        {type_label: {count, pct, avg_isr_clean, avg_isr_noisy, avg_nar_noisy,
                       avg_nar_corrected, avg_f1_clean, avg_f1_noisy, avg_f1_corrected,
                       contains_rate_clean, contains_rate_noisy, contains_rate_corrected}}
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        grouped[c["type"]].append(c)

    stats: dict[str, dict] = {}
    total = len(cases)
    for tn in TYPE_ORDER:
        items = grouped.get(tn, [])
        if not items:
            continue
        n = len(items)

        def _safe_mean(key: str) -> float | None:
            vals = [it.get(key) for it in items if it.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _safe_rate(key: str) -> float | None:
            vals = [it.get(key) for it in items if it.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        stats[tn] = {
            "count": n,
            "pct": round(n / total * 100, 1) if total else 0.0,
            "avg_f1_clean": _safe_mean("f1_clean"),
            "avg_f1_noisy": _safe_mean("f1_noisy"),
            "avg_f1_corrected": _safe_mean("f1_corrected"),
            "avg_isr_clean": _safe_mean("isr_clean"),
            "avg_isr_noisy": _safe_mean("isr_noisy"),
            "avg_nar_noisy": _safe_mean("nar_noisy"),
            "avg_nar_corrected": _safe_mean("nar_corrected"),
            "contains_rate_clean": _safe_rate("contains_clean"),
            "contains_rate_noisy": _safe_rate("contains_noisy"),
            "contains_rate_corrected": _safe_rate("contains_corrected"),
        }
    return stats


def _compute_transition_matrix(
    cases_by_ratio: dict[float, dict[int, str]],
    ratios: list[float],
) -> list[dict]:
    """计算相邻比例之间的案例类型过渡矩阵。

    返回形如 [{"from_ratio": r1, "to_ratio": r2, "matrix": {typeA: {typeB: count}}}] 的列表。
    """
    transitions: list[dict] = []
    for i in range(len(ratios) - 1):
        r_from = ratios[i]
        r_to = ratios[i + 1]
        mat: dict[str, Counter] = defaultdict(Counter)
        types_from = cases_by_ratio.get(r_from, {})
        types_to = cases_by_ratio.get(r_to, {})
        for sid, t_from in types_from.items():
            t_to = types_to.get(sid)
            if t_to is not None:
                mat[t_from][t_to] += 1
        serializable = {
            tn: dict(cnt) for tn, cnt in mat.items()
        }
        transitions.append({
            "from_ratio": r_from,
            "to_ratio": r_to,
            "matrix": serializable,
        })
    return transitions


def _index_by_sample(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        out[r["sample_id"]] = r
    return out


def _pick_display_cases(cases: list[dict], *, k: int) -> list[dict]:
    """从全量分类中按优先级排序，取 top-K 供报告展示。"""
    priority = {
        "Type1-矫正生效": 0,
        "Type2-噪音激发": 1,
        "Other": 2,
        "Type1-矫正未生效": 3,
        "Type3-信息淹没": 4,
        "Type4-免疫": 5,
    }
    cases.sort(key=lambda x: (priority.get(x["type"], 5), x["sample_id"]))
    return cases[:k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=0,
                   help="候选样本池大小（0 = 使用全部数据）")
    p.add_argument("--pick", type=int, default=30,
                   help="每个 ratio 挑选用于报告展示的案例数（分类统计仍基于全量）")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--subset", default="main",
                   choices=("main", "refine", "fact", "int"))
    p.add_argument(
        "--ratios",
        default="0.5,0.75,1.0",
        help="噪音比例列表（逗号分隔），默认 0.5,0.75,1.0",
    )
    p.add_argument(
        "--corrector",
        default="confidence",
        help="用于案例矫正展示的方法名（默认 confidence）",
    )
    args = p.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    limit = args.n if args.n > 0 else None

    records = load_corpus(language=args.language, subset=args.subset, limit=limit)
    logger.info(f"exp3 (large): {len(records)} samples × (1 clean + {len(ratios)}×2) = {len(records) * (1 + len(ratios) * 2)} LLM calls max")

    # clean 只跑一次，所有 ratio 共享
    conditions = [
        RunCondition(method="naive", noise_ratio=0.0, label="clean"),
    ]
    for r in ratios:
        conditions.append(
            RunCondition(
                method="naive", noise_ratio=r, noise_type="semantic",
                label=f"noisy_r{r}",
            )
        )
        conditions.append(
            RunCondition(
                method=args.corrector, noise_ratio=r, noise_type="semantic",
                label=f"corrected_r{r}",
            )
        )

    results = run_conditions(records=records, conditions=conditions, language=args.language)

    # clean 是第 0 个 result
    clean_rows = _index_by_sample(results[0].rows)

    # ── 全量分类 + 统计 ──
    all_cases: list[dict] = []
    display_cases: list[dict] = []
    ratio_summaries: dict[float, dict] = {}
    cases_by_ratio: dict[float, dict[int, str]] = {}

    for i, r in enumerate(ratios):
        noisy_rows = _index_by_sample(results[1 + i * 2].rows)
        corr_rows = _index_by_sample(results[1 + i * 2 + 1].rows)

        # 全量分类
        cases = _classify_all_cases(clean_rows, noisy_rows, corr_rows, ratio=r)
        all_cases.extend(cases)

        # 全量 type 分布
        type_counts = Counter(c["type"] for c in cases)
        per_type_stats = _compute_per_type_stats(cases)

        # 记录每个样本的类型（用于过渡矩阵）
        cases_by_ratio[r] = {c["sample_id"]: c["type"] for c in cases}

        # 挑选 top-K 用于报告展示
        picked = _pick_display_cases(list(cases), k=args.pick)
        display_cases.extend(picked)

        ratio_summaries[r] = {
            "n_total": len(cases),
            "type_distribution": dict(type_counts),
            "type_proportions": {
                tn: round(type_counts.get(tn, 0) / len(cases) * 100, 1) if cases else 0.0
                for tn in TYPE_ORDER
            },
            "per_type_stats": per_type_stats,
            "display_cases": picked,
        }

        if cases:
            dist_str = ", ".join(
                f"{tn}={type_counts.get(tn,0)}({ratio_summaries[r]['type_proportions'][tn]}%)"
                for tn in TYPE_ORDER if type_counts.get(tn, 0) > 0
            )
        else:
            dist_str = "no cases"
        print(f"  r={r}: {len(cases)} cases total | {dist_str}")

    # ── 跨比例过渡矩阵 ──
    transition_matrix = _compute_transition_matrix(cases_by_ratio, ratios)
    if transition_matrix:
        print("\nCross-ratio transitions:")
        for tm in transition_matrix:
            print(f"  r={tm['from_ratio']} -> r={tm['to_ratio']}:")
            for t_from, t_to_counts in tm["matrix"].items():
                items = ", ".join(f"{t}=>{n}" for t, n in t_to_counts.items() if n > 0)
                print(f"    {t_from}: {items}")

    extras: dict[str, Any] = {
        "cases": all_cases,
        "display_cases": display_cases,
        "ratio_summaries": ratio_summaries,
        "transition_matrix": transition_matrix,
        "n_ratios": len(ratios),
        "total_cases": len(all_cases),
        "n_samples": len(records),
        "args": vars(args),
    }
    path = save_run(
        experiment_name=f"exp3_case_study_{args.language}_{args.subset}",
        results=results,
        extras=extras,
    )
    print(f"\nsaved -> {path}")
    print(f"total: {len(all_cases)} classified cases across {len(ratios)} ratios "
          f"({len(display_cases)} picked for display)")


if __name__ == "__main__":
    main()
