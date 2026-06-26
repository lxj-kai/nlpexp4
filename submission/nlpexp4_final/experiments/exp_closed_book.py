"""Closed-book vs Oracle RAG 对照实验。

在同一批样本上比较：
- closed_book：只给问题，不提供任何文档
- naive @ r=0：提供 gold supporting 文档（Oracle RAG）

数据集：CmedqaRetrieval (zh) + MIRIAD (en) + 2WikiMultihopQA (en)

用法:
    python -m experiments.exp_closed_book --n 50
    python -m experiments.exp_closed_book --n 100 --workers 5
"""
from __future__ import annotations

import argparse

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.utils import get_logger

logger = get_logger("exp_closed_book")

DATASETS = (
    ("cmedqa", "zh", "main"),
    ("miriad", "en", "main"),
    ("2wiki", "en", "main"),
)

CONDITIONS = (
    RunCondition(
        method="closed_book",
        noise_ratio=0.0,
        noise_type="semantic",
        noise_position="interleave",
        label="closed_book",
    ),
    RunCondition(
        method="naive",
        noise_ratio=0.0,
        noise_type="semantic",
        noise_position="interleave",
        label="naive|r=0|oracle_rag",
    ),
)


def _run_one(
    *,
    dataset: str,
    language: str,
    subset: str,
    n: int,
    workers: int,
) -> str:
    records = load_corpus(
        language=language, subset=subset, dataset=dataset, limit=n  # type: ignore[arg-type]
    )
    logger.info(
        f"closed_book {dataset}/{language}/{subset}: "
        f"{len(records)} samples × {len(CONDITIONS)} conditions"
    )
    results = run_conditions(
        records=records,
        conditions=list(CONDITIONS),
        language=language,
        show_progress=True,
        workers=workers,
    )
    comparison = {}
    for r in results:
        comparison[r.condition.method] = {
            "judge_score": r.summary.get("judge_score"),
            "judge_correct": r.summary.get("judge_correct"),
            "isr": r.summary.get("isr"),
            "nar": r.summary.get("nar"),
        }
    closed = comparison.get("closed_book", {})
    rag = comparison.get("naive", {})
    delta_judge = None
    if closed.get("judge_score") is not None and rag.get("judge_score") is not None:
        delta_judge = round(float(rag["judge_score"]) - float(closed["judge_score"]), 4)

    return save_run(
        experiment_name=f"exp_closed_book_{dataset}_{language}_{subset}",
        results=results,
        extras={
            "args": {
                "dataset": dataset,
                "language": language,
                "subset": subset,
                "n": n,
                "workers": workers,
            },
            "comparison": {
                "closed_book": closed,
                "oracle_rag_naive_r0": rag,
                "delta_judge_rag_minus_closed": delta_judge,
            },
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-book vs Oracle RAG on 3 datasets")
    p.add_argument("--n", type=int, default=50, help="每个数据集的样本数")
    p.add_argument("--workers", type=int, default=5, help="并行 worker 数")
    args = p.parse_args()

    paths: list[str] = []
    for dataset, language, subset in DATASETS:
        paths.append(
            _run_one(
                dataset=dataset,
                language=language,
                subset=subset,
                n=args.n,
                workers=args.workers,
            )
        )

    print("\n=== Closed-book vs Oracle RAG ===")
    for path in paths:
        print(f"  {path}")
    print("\nRun: python scripts/analyze_closed_book_results.py")


if __name__ == "__main__":
    main()
