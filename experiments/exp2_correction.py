"""实验二：矫正方法对比。

矩阵：
- 方法 ∈ {naive, prompt, iterative, confidence, selfrag}
- 噪音比例 ∈ {0.0(clean baseline), 0.25, 0.5, 0.75}
- 噪音类型固定为 semantic（与 PLAN 表 2 对齐）

用法：
    python -m experiments.exp2_correction --n 50 --language zh
    python -m experiments.exp2_correction --n 100 --methods naive,prompt,confidence
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
from src.correctors import list_correctors
from src.utils import get_logger

logger = get_logger("exp2")

DEFAULT_METHODS = ("naive", "prompt", "iterative", "confidence", "selfrag")
DEFAULT_RATIOS = (0.0, 0.25, 0.5, 0.75)


def build_conditions(methods, ratios, noise_type: str = "semantic") -> list[RunCondition]:
    return [
        RunCondition(
            method=m,
            noise_ratio=r,
            noise_type=noise_type,
            noise_position="interleave",
            label=f"{m}|r={r}|{noise_type}",
        )
        for m, r in product(methods, ratios)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=CONFIG.smoke_test_size)
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--subset", default="main", choices=("main", "refine", "fact", "int"))
    p.add_argument(
        "--noise-type", default="semantic", choices=("semantic", "counterfactual", "mixed"),
        help="噪音类型（默认 semantic）",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="方法列表（逗号分隔），可选: " + ",".join(["naive", *list_correctors()]),
    )
    p.add_argument(
        "--ratios",
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help="噪音比例列表（逗号分隔）",
    )
    args = p.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    ratios = tuple(float(r) for r in args.ratios.split(",") if r.strip())

    records = load_corpus(language=args.language, subset=args.subset, limit=args.n)
    conditions = build_conditions(methods, ratios, noise_type=args.noise_type)
    logger.info(f"exp2: {len(records)} samples × {len(conditions)} conditions")
    results = run_conditions(records=records, conditions=conditions, language=args.language)
    path = save_run(
        experiment_name=f"exp2_correction_{args.language}_{args.subset}_{args.noise_type}",
        results=results,
        extras={"args": vars(args)},
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
