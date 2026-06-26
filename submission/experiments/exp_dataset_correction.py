"""TEMPO / MultiHop-RAG 正式矫正对比实验（exp2 矩阵 + 可视化）。

在 sanity n=100 naive-only 基线之上，补齐方法 × 噪音比例对比。

用法:
    python -m experiments.exp_dataset_correction --n 50
    python -m experiments.exp_dataset_correction --n 50 --dataset tempo
    python -m experiments.exp_dataset_correction --n 2 --dry
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.smoke_test import DEFAULT_METHODS, DEFAULT_RATIOS, run as batch_run
from src.utils import get_logger

logger = get_logger("exp_dataset_correction")

DATASETS = (
    ("tempo", "en", "main"),
    ("multihop_rag", "en", "main"),
)


def main() -> None:
    p = argparse.ArgumentParser(description="TEMPO / MultiHop-RAG 矫正对比")
    p.add_argument("--n", type=int, default=50, help="每数据集样本数")
    p.add_argument(
        "--dataset",
        choices=("tempo", "multihop_rag", "all"),
        default="all",
        help="目标数据集（默认两者都跑）",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="矫正方法（逗号分隔）",
    )
    p.add_argument(
        "--ratios",
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help="噪音比例（逗号分隔）",
    )
    p.add_argument("--noise-type", default="semantic")
    p.add_argument("--noise-position", default="interleave")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry", action="store_true", help="离线接线，不调用 LLM")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--figures-dir", default=None)
    args = p.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    ratios = tuple(float(r) for r in args.ratios.split(",") if r.strip())

    targets = DATASETS
    if args.dataset != "all":
        targets = tuple(t for t in DATASETS if t[0] == args.dataset)

    paths: list[str] = []
    for dataset, language, subset in targets:
        logger.info("=" * 60)
        logger.info(f"exp_correction: {dataset}/{language}/{subset} n={args.n}")
        out = batch_run(
            n=args.n,
            dry=args.dry,
            language=language,
            subset=subset,
            dataset=dataset,
            methods=methods,
            ratios=ratios,
            noise_type=args.noise_type,
            noise_position=args.noise_position,
            workers=args.workers,
            render_figures=not args.no_figures,
            figures_dir=args.figures_dir,
            experiment_prefix="exp_correction",
        )
        paths.append(out["result_json"])

    print("saved:")
    for path in paths:
        print(f"  {path}")
        if not args.no_figures and not args.dry:
            print(f"    figures -> figures/{Path(path).stem}/")


if __name__ == "__main__":
    main()
