"""实验一：噪音影响分析（核心实验）。

矩阵：
- 噪音比例 ∈ {0, 0.25, 0.5, 0.75, 1.0}
- 噪音类型 ∈ {semantic, counterfactual, mixed}
- 数据集 ∈ {zh, en}  (--language 切换)
- 位置子实验：在 ratio=0.5 下跑 {front, back, interleave, surround}

用法：
    python -m experiments.exp1_noise_impact --n 50 --language zh
    python -m experiments.exp1_noise_impact --n 300 --language zh --position
"""
from __future__ import annotations

import argparse
from itertools import product

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.config import CONFIG
from src.utils import get_logger

logger = get_logger("exp1")


def build_main_conditions() -> list[RunCondition]:
    out: list[RunCondition] = []
    for ratio, ntype in product(CONFIG.noise_ratios, CONFIG.noise_types):
        out.append(
            RunCondition(
                method="naive",
                noise_ratio=ratio,
                noise_type=ntype,
                noise_position="interleave",
                label=f"naive|r={ratio}|{ntype}",
            )
        )
    return out


def build_position_conditions() -> list[RunCondition]:
    return [
        RunCondition(
            method="naive",
            noise_ratio=0.5,
            noise_type="semantic",
            noise_position=pos,
            label=f"naive|pos={pos}",
        )
        for pos in CONFIG.noise_positions
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=CONFIG.smoke_test_size, help="样本数")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument(
        "--position", action="store_true", help="是否额外跑位置子实验（默认不跑，省 token）"
    )
    p.add_argument(
        "--position-only",
        action="store_true",
        help="仅跑位置子实验（4 conditions），跳过主矩阵以节省 token",
    )
    p.add_argument("--subset", default="main", choices=("main", "refine", "fact", "int"))
    args = p.parse_args()

    records = load_corpus(language=args.language, subset=args.subset, limit=args.n)
    if args.position_only:
        conditions = build_position_conditions()
        tag_suffix = "_position"
    else:
        conditions = build_main_conditions()
        if args.position:
            conditions += build_position_conditions()
        tag_suffix = ""

    logger.info(f"exp1: {len(records)} samples × {len(conditions)} conditions")
    results = run_conditions(records=records, conditions=conditions, language=args.language)
    path = save_run(
        experiment_name=f"exp1_noise_impact_{args.language}_{args.subset}{tag_suffix}",
        results=results,
        extras={"args": vars(args)},
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
