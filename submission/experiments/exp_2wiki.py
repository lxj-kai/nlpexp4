"""2WikiMultihopQA 完整实验套件。

在 hard-negative 多跳 QA 上复现核心噪音实验、矫正对比与现有方法横向比较。

用法:
    python -m experiments.exp_2wiki --n 500 --workers 10
    python -m experiments.exp_2wiki --n 500 --workers 10 --phase exp1
    bash scripts/run_2wiki_n500.sh
"""
from __future__ import annotations

import argparse
from itertools import product

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.config import CONFIG
from src.utils import get_logger

logger = get_logger("exp_2wiki")

DATASET = "2wiki"
LANGUAGE = "en"

EXP1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXP2_RATIOS = (0.0, 0.25, 0.5, 0.75)
EXP2_METHODS = ("naive", "prompt", "iterative", "confidence", "voting")
EXP4_RATIOS = (0.5, 0.75)
EXP4_METHODS = ("naive", "selfrag", "iterative", "confidence", "prompt", "voting")


def _exp1_conditions(*, noise_type: str = "semantic") -> list[RunCondition]:
    return [
        RunCondition(
            method="naive",
            noise_ratio=ratio,
            noise_type=noise_type,  # type: ignore[arg-type]
            noise_position="interleave",
            label=f"naive|r={ratio}|{noise_type}",
        )
        for ratio in EXP1_RATIOS
    ]


def _exp2_conditions(*, noise_type: str = "semantic") -> list[RunCondition]:
    return [
        RunCondition(
            method=method,
            noise_ratio=ratio,
            noise_type=noise_type,  # type: ignore[arg-type]
            noise_position="interleave",
            label=f"{method}|r={ratio}",
        )
        for method, ratio in product(EXP2_METHODS, EXP2_RATIOS)
    ]


def _exp4_conditions(*, noise_type: str = "semantic") -> list[RunCondition]:
    return [
        RunCondition(
            method=method,
            noise_ratio=ratio,
            noise_type=noise_type,  # type: ignore[arg-type]
            noise_position="interleave",
            label=f"{method}|r={ratio}|{noise_type}",
        )
        for method, ratio in product(EXP4_METHODS, EXP4_RATIOS)
    ]


def _run_phase(
    *,
    phase: str,
    subset: str,
    n: int,
    conditions: list[RunCondition],
    noise_type: str,
    workers: int,
) -> str:
    records = load_corpus(
        language=LANGUAGE,
        subset=subset,  # type: ignore[arg-type]
        dataset=DATASET,
        limit=n,
    )
    logger.info(
        f"{phase} {DATASET}/{LANGUAGE}/{subset}: "
        f"{len(records)} samples × {len(conditions)} conditions"
    )
    results = run_conditions(
        records=records,
        conditions=conditions,
        language=LANGUAGE,
        dataset=DATASET,
        show_progress=True,
        workers=workers,
    )
    name = f"exp_2wiki_{phase}_{LANGUAGE}_{subset}"
    return save_run(
        experiment_name=name,
        results=results,
        extras={
            "args": {
                "phase": phase,
                "dataset": DATASET,
                "language": LANGUAGE,
                "subset": subset,
                "noise_type": noise_type,
                "n": n,
                "workers": workers,
            }
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="2WikiMultihopQA full experiment suite")
    p.add_argument("--n", type=int, default=500, help="每条件样本数")
    p.add_argument("--workers", type=int, default=10, help="样本级并发线程数")
    p.add_argument(
        "--phase",
        choices=("all", "exp1", "exp1_fact", "exp2", "exp4"),
        default="all",
        help="运行阶段（可并行分进程跑不同 phase）",
    )
    args = p.parse_args()

    jobs: list[tuple[str, str, str, list[RunCondition]]] = []
    if args.phase in ("all", "exp1"):
        jobs.append(("exp1", "main", "semantic", _exp1_conditions(noise_type="semantic")))
    if args.phase in ("all", "exp1_fact"):
        jobs.append(
            ("exp1_fact", "fact", "counterfactual", _exp1_conditions(noise_type="counterfactual"))
        )
    if args.phase in ("all", "exp2"):
        jobs.append(("exp2", "main", "semantic", _exp2_conditions(noise_type="semantic")))
    if args.phase in ("all", "exp4"):
        jobs.append(("exp4", "main", "semantic", _exp4_conditions(noise_type="semantic")))

    paths: list[str] = []
    for phase, subset, noise_type, conditions in jobs:
        paths.append(
            _run_phase(
                phase=phase,
                subset=subset,
                n=args.n,
                conditions=conditions,
                noise_type=noise_type,
                workers=args.workers,
            )
        )

    print("saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
