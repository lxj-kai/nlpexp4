"""三数据集噪音梯度 + 矫正恢复实验。

目标：验证 低噪音≈能答、中噪音≈勉强、高噪音≈不能答，且矫正后部分恢复。

数据集（各 n=500）：
  - rgb      zh/main
  - cmedqa   zh/main
  - 2wiki    en/main

条件：
  - noise_ratio ∈ {0.0, 0.5, 0.75}  （低 / 中 / 高）
  - noise_type = semantic
  - methods ∈ {naive, prompt, confidence}

用法：
  python -m experiments.exp_noise_gradient --n 500
  python -m experiments.exp_noise_gradient --n 50 --datasets rgb   # 快速试跑
"""
from __future__ import annotations

import argparse
from itertools import product

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.evaluator import Evaluator
from src.llm_client import LLMClient, get_judge_client
from src.utils import get_logger

logger = get_logger("exp_noise_gradient")

DATASET_SPECS: dict[str, tuple[str, str, str, int | None]] = {
    # (dataset, language, subset, max_available hint)
    "rgb": ("rgb", "zh", "main", 300),
    "cmedqa": ("cmedqa", "zh", "main", None),
    "2wiki": ("2wiki", "en", "main", None),
}

RATIOS = (0.0, 0.5, 0.75)
METHODS = ("naive", "prompt", "confidence")


def build_conditions(methods: tuple[str, ...], ratios: tuple[float, ...]) -> list[RunCondition]:
    return [
        RunCondition(
            method=m,
            noise_ratio=r,
            noise_type="semantic",
            noise_position="interleave",
            label=f"{m}|r={r}",
        )
        for m, r in product(methods, ratios)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument(
        "--datasets",
        default="rgb,cmedqa,2wiki",
        help="逗号分隔：rgb,cmedqa,2wiki",
    )
    p.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="方法列表（逗号分隔）",
    )
    p.add_argument(
        "--ratios",
        default=",".join(str(r) for r in RATIOS),
        help="噪音比例（逗号分隔）",
    )
    p.add_argument("--no-figures", action="store_true", help="跳过可视化出图")
    p.add_argument("--figures-dir", default=None, help="图表输出目录")
    args = p.parse_args()

    ds_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    ratios = tuple(float(r) for r in args.ratios.split(",") if r.strip())
    conditions = build_conditions(methods, ratios)

    llm = LLMClient()
    judge = get_judge_client()
    evaluator = Evaluator(
        use_llm_judge=True,
        use_legacy_metrics=False,
        use_semantic_attribution=False,
        llm=llm,
        judge_llm=judge,
    )

    saved: list[str] = []
    for key in ds_keys:
        if key not in DATASET_SPECS:
            raise SystemExit(f"unknown dataset {key!r}; choose from {list(DATASET_SPECS)}")
        dataset, language, subset, cap = DATASET_SPECS[key]
        n = args.n if cap is None else min(args.n, cap)
        if cap is not None and args.n > cap:
            logger.warning(f"{key} 仅 {cap} 条，使用 n={n}（请求 {args.n}）")
        logger.info("=" * 60)
        logger.info(f"dataset={key} n={n} lang={language} conditions={len(conditions)}")
        records = load_corpus(
            language=language,  # type: ignore[arg-type]
            subset=subset,  # type: ignore[arg-type]
            dataset=dataset,  # type: ignore[arg-type]
            limit=n,
        )
        results = run_conditions(
            records=records,
            conditions=conditions,
            llm=llm,
            evaluator=evaluator,
            language=language,
            dataset=dataset,
            show_progress=True,
        )
        path = save_run(
            experiment_name=f"exp_noise_gradient_{key}_n{n}",
            results=results,
            extras={
                "args": vars(args),
                "dataset_key": key,
                "dataset": dataset,
                "language": language,
                "subset": subset,
            },
        )
        saved.append(path)
        print(f"saved {key} -> {path}")
        if not args.no_figures:
            from src.visualize import render_batch_run_figures

            figs = render_batch_run_figures(path, out_dir=args.figures_dir, tag=f"{key}_n{n}")
            for fp in figs:
                print(f"  figure -> {fp}")

    print("\nAll done:")
    for pth in saved:
        print(f"  {pth}")
    print("\nRun analysis:")
    print("  python scripts/analyze_noise_gradient.py " + " ".join(saved))


if __name__ == "__main__":
    main()
