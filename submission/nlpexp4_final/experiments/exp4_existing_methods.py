"""实验四：与现有方法对比。

横向比较 5 个方法：
- naive (基线)
- selfrag (Self-RAG 思路)
- iterative (CRAG 分解-重组思路)
- confidence (CoT-RAG 链式推理思路)
- prompt (本工作主方法 A — 仅 prompt 改造)

固定 ratio=0.5, type=semantic（最具代表性的中等强度噪音）。

用法：
    python -m experiments.exp4_existing_methods --n 100 --language zh
"""
from __future__ import annotations

import argparse

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.config import CONFIG
from src.utils import get_logger

logger = get_logger("exp4")


DEFAULT_METHODS = ("naive", "selfrag", "iterative", "confidence", "prompt", "voting")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=CONFIG.smoke_test_size)
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument(
        "--subset", default="main", choices=("main", "refine", "fact", "int")
    )
    p.add_argument(
        "--ratio", type=float, default=0.5, help="噪音比例（默认 0.5）"
    )
    p.add_argument(
        "--noise-type",
        choices=("semantic", "counterfactual", "mixed"),
        default="semantic",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="方法列表，逗号分隔",
    )
    args = p.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    records = load_corpus(language=args.language, subset=args.subset, limit=args.n)
    conditions = [
        RunCondition(
            method=m,
            noise_ratio=args.ratio,
            noise_type=args.noise_type,
            noise_position="interleave",
            label=m,
        )
        for m in methods
    ]
    logger.info(f"exp4: {len(records)} samples × {len(methods)} methods on {args.subset}/{args.noise_type}")
    results = run_conditions(records=records, conditions=conditions, language=args.language)
    path = save_run(
        experiment_name=f"exp4_existing_methods_{args.language}_{args.subset}_{args.noise_type}",
        results=results,
        extras={"args": vars(args)},
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
