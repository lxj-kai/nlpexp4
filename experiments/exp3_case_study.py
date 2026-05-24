"""实验三：案例深度分析。

针对挑选出的若干样本，同时跑 clean / noisy / corrected，
保存"问题-参考答案-3 种输出-文档-误导路径"五元组，便于报告里逐条展示。

候选策略：
1. 用户指定 sample_ids；或
2. 自动从指定数据集前 N 条中随机抽 K 条；或
3. 跑完后按"clean 对 → noisy 错"自动挑出 K 个 Type1/2/3 典型样本。

用法：
    python -m experiments.exp3_case_study --n 30 --pick 15
"""
from __future__ import annotations

import argparse
from typing import Any

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.config import CONFIG
from src.evaluator import _exact_match, _contains_match
from src.utils import get_logger

logger = get_logger("exp3")


_TYPE_PRIORITY = {
    "Type1-矫正生效": 0,
    "Type1-矫正未生效": 1,
    "Type2-噪音激发": 2,
    "Type4-免疫": 3,
    "Type3-淹没": 4,
    "Other": 9,
}


def _classify(c_ok: bool, n_ok: bool, cor_ok: bool | None) -> str:
    """Type1: clean对→noisy错（关心矫正能否救回来）
    Type2: clean错→noisy对（"噪音激发"，研究反直觉现象）
    Type3: clean错→noisy错（双错淹没）
    Type4: clean对→noisy对（噪音免疫）
    """
    if c_ok and not n_ok:
        if cor_ok is True:
            return "Type1-矫正生效"
        return "Type1-矫正未生效"
    if not c_ok and n_ok:
        return "Type2-噪音激发"
    if c_ok and n_ok:
        return "Type4-免疫"
    if not c_ok and not n_ok:
        return "Type3-淹没"
    return "Other"


def _pick_typical_cases(
    naive_clean: dict, naive_noisy: dict, corrected: dict, *, k: int
) -> list[dict]:
    """匹配 clean / noisy / corrected 三组，按 Type1–4 分类后按优先级挑前 k 个。"""
    cases: list[dict] = []
    for sid, c in naive_clean.items():
        n = naive_noisy.get(sid)
        if n is None:
            continue
        cor = corrected.get(sid)
        c_ok = bool(c.get("contains"))
        n_ok = bool(n.get("contains"))
        cor_ok = bool(cor.get("contains")) if cor else None
        type_label = _classify(c_ok, n_ok, cor_ok)
        cases.append(
            {
                "sample_id": sid,
                "type": type_label,
                "query": c.get("query", ""),
                "gold": c.get("gold"),
                "pred_clean": c.get("prediction"),
                "pred_noisy": n.get("prediction"),
                "pred_corrected": cor.get("prediction") if cor else None,
                "contains_clean": c_ok,
                "contains_noisy": n_ok,
                "contains_corrected": cor_ok,
                "isr_clean": c.get("isr"),
                "isr_noisy": n.get("isr"),
                "nar_noisy": n.get("nar"),
                "nar_corrected": cor.get("nar") if cor else None,
            }
        )
    cases.sort(
        key=lambda x: (_TYPE_PRIORITY.get(x["type"], 99), x["sample_id"])
    )
    return cases[:k]


def _index_by_sample(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        out[r["sample_id"]] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="候选样本池大小")
    p.add_argument("--pick", type=int, default=15, help="最终挑选案例数")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument(
        "--corrector",
        default="confidence",
        help="用于案例矫正展示的方法名（默认 confidence）",
    )
    args = p.parse_args()

    records = load_corpus(language=args.language, subset="main", limit=args.n)
    conditions = [
        RunCondition(method="naive", noise_ratio=0.0, label="clean"),
        RunCondition(
            method="naive", noise_ratio=0.5, noise_type="semantic", label="noisy"
        ),
        RunCondition(
            method=args.corrector,
            noise_ratio=0.5,
            noise_type="semantic",
            label=f"corrected_{args.corrector}",
        ),
    ]
    results = run_conditions(records=records, conditions=conditions, language=args.language)

    clean_rows = _index_by_sample(results[0].rows)
    noisy_rows = _index_by_sample(results[1].rows)
    corr_rows = _index_by_sample(results[2].rows)
    cases = _pick_typical_cases(clean_rows, noisy_rows, corr_rows, k=args.pick)

    extras: dict[str, Any] = {
        "cases": cases,
        "args": vars(args),
    }
    path = save_run(
        experiment_name=f"exp3_case_study_{args.language}",
        results=results,
        extras=extras,
    )
    print(f"saved -> {path}")
    print(f"picked {len(cases)} cases by type")


if __name__ == "__main__":
    main()
