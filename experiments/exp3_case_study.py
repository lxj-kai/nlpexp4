"""实验三：案例深度分析。

针对挑选出的若干样本，同时跑 clean / noisy / corrected，
在多个噪音比例 (0.5-1.0) 下循环，保存分类案例便于报告逐条展示。

用法：
    python -m experiments.exp3_case_study --n 30 --pick 20
    python -m experiments.exp3_case_study --n 50 --pick 30 --ratios 0.5,0.75,1.0
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.utils import get_logger

logger = get_logger("exp3")


def _pick_typical_cases(
    naive_clean: dict, naive_noisy: dict, corrected: dict, *, k: int, ratio: float
) -> list[dict]:
    """匹配 clean / noisy / corrected 三组，按 Type1-4 分类。"""
    cases: list[dict] = []
    for sid, c in naive_clean.items():
        n = naive_noisy.get(sid)
        cor = corrected.get(sid)
        if n is None:
            continue
        type_label: str
        if c.get("contains") and not n.get("contains"):
            if cor and cor.get("contains"):
                type_label = "Type1-矫正生效"
            else:
                type_label = "Type1-矫正未生效"
        elif c.get("contains") and n.get("contains"):
            type_label = "Type4-免疫"
        elif not c.get("contains") and not n.get("contains"):
            type_label = "Type3-信息淹没"
        else:
            type_label = "Other"
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
            }
        )
    # 优先 Type1(矫正生效) > Other > Type1(未生效) > Type3 > Type4
    priority = {
        "Type1-矫正生效": 0,
        "Other": 1,
        "Type1-矫正未生效": 2,
        "Type3-信息淹没": 3,
        "Type4-免疫": 4,
    }
    cases.sort(key=lambda x: (priority.get(x["type"], 5), x["sample_id"]))
    return cases[:k]


def _index_by_sample(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        out[r["sample_id"]] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="候选样本池大小")
    p.add_argument("--pick", type=int, default=20, help="每个 ratio 挑选案例数")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
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

    records = load_corpus(language=args.language, subset="main", limit=args.n)

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

    logger.info(f"exp3: {len(records)} samples × {len(conditions)} conditions (clean + {len(ratios)}×2 per ratio)")
    results = run_conditions(records=records, conditions=conditions, language=args.language)

    # clean 是第 0 个 result
    clean_rows = _index_by_sample(results[0].rows)

    # 每个 ratio 取对应的 noisy + corrected results
    all_cases: list[dict] = []
    ratio_summaries: dict[float, dict] = {}
    for i, r in enumerate(ratios):
        noisy_rows = _index_by_sample(results[1 + i * 2].rows)
        corr_rows = _index_by_sample(results[1 + i * 2 + 1].rows)
        cases = _pick_typical_cases(clean_rows, noisy_rows, corr_rows, k=args.pick, ratio=r)
        all_cases.extend(cases)
        type_counts = Counter(c["type"] for c in cases)
        ratio_summaries[r] = {
            "n_cases": len(cases),
            "type_distribution": dict(type_counts),
        }
        print(f"  r={r}: picked {len(cases)} cases, dist={dict(type_counts)}")

    extras: dict[str, Any] = {
        "cases": all_cases,
        "ratio_summaries": ratio_summaries,
        "n_ratios": len(ratios),
        "total_cases": len(all_cases),
        "args": vars(args),
    }
    path = save_run(
        experiment_name=f"exp3_case_study_{args.language}",
        results=results,
        extras=extras,
    )
    print(f"\nsaved -> {path}")
    print(f"total: {len(all_cases)} cases across {len(ratios)} ratios")


if __name__ == "__main__":
    main()
