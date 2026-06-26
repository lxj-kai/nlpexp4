"""新数据集基准实验：MIRIAD (en) + CmedqaRetrieval (zh)。

在完整规模数据集上复现核心噪音实验与矫正对比，用于与 RGB 小样本结果对照。

用法:
    python -m experiments.exp_new_datasets --n 25
    python -m experiments.exp_new_datasets --full          # Cmedqa 3999 + MIRIAD 3999
    python -m experiments.exp_new_datasets --n-cmedqa 3999 --n-miriad 3000
    python -m experiments.exp_new_datasets --n 100 --phase exp1
"""
from __future__ import annotations

import argparse
import json
from itertools import product

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.config import CONFIG, PROJECT_ROOT
from src.utils import get_logger

logger = get_logger("exp_new_datasets")

EXP1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXP2_RATIOS = (0.0, 0.5, 0.75)
EXP2_METHODS = ("naive", "confidence")

# Cmedqa 优先：体量小、可先出结果；MIRIAD 全库 580 万条
DATASETS = (
    ("cmedqa", "zh", "main"),
    ("miriad", "en", "main"),
)

_DATASET_CAPS: dict[str, int] = {
    "cmedqa": 3_999,
    "miriad": 5_821_948,
}


def _cap_from_metadata(dataset: str) -> int | None:
    meta_path = PROJECT_ROOT / "data" / dataset / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if dataset == "cmedqa":
        return int(meta.get("main_size") or 0) or None
    return _DATASET_CAPS.get(dataset)


def resolve_sample_size(dataset: str, n: int | None, *, full: bool = False) -> int:
    cap = _cap_from_metadata(dataset) or _DATASET_CAPS[dataset]
    if full:
        target = _cap_from_metadata("cmedqa") or 3_999
        return min(cap, target)
    if n is None:
        raise ValueError(f"sample size required for {dataset}; pass --n or --full")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > cap:
        logger.warning(f"{dataset}: requested n={n} exceeds cap {cap}, using {cap}")
        return cap
    return n


def _exp1_conditions() -> list[RunCondition]:
    return [
        RunCondition(
            method="naive",
            noise_ratio=ratio,
            noise_type="semantic",
            noise_position="interleave",
            label=f"naive|r={ratio}|semantic",
        )
        for ratio in EXP1_RATIOS
    ]


def _exp2_conditions() -> list[RunCondition]:
    return [
        RunCondition(
            method=method,
            noise_ratio=ratio,
            noise_type="semantic",
            noise_position="interleave",
            label=f"{method}|r={ratio}",
        )
        for method, ratio in product(EXP2_METHODS, EXP2_RATIOS)
    ]


def _run_one(
    *,
    phase: str,
    dataset: str,
    language: str,
    subset: str,
    n: int,
    conditions: list[RunCondition],
) -> str:
    records = load_corpus(
        language=language, subset=subset, dataset=dataset, limit=n  # type: ignore[arg-type]
    )
    logger.info(
        f"{phase} {dataset}/{language}/{subset}: "
        f"{len(records)} samples × {len(conditions)} conditions"
    )
    results = run_conditions(
        records=records,
        conditions=conditions,
        language=language,
        dataset=dataset,
        show_progress=True,
    )
    name = f"exp_new_{phase}_{dataset}_{language}_{subset}"
    return save_run(
        experiment_name=name,
        results=results,
        extras={
            "args": {
                "phase": phase,
                "dataset": dataset,
                "language": language,
                "subset": subset,
                "n": n,
            }
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark MIRIAD + CmedqaRetrieval")
    p.add_argument("--n", type=int, default=None, help="两数据集统一样本数（可被 --n-* 覆盖）")
    p.add_argument("--n-cmedqa", type=int, default=None, help="CmedqaRetrieval 样本数（上限 3999）")
    p.add_argument("--n-miriad", type=int, default=None, help="MIRIAD 样本数（上限 582 万）")
    p.add_argument(
        "--full",
        action="store_true",
        help="大规模模式：Cmedqa 全量 3999 + MIRIAD 3999（与中文集对齐）",
    )
    p.add_argument(
        "--phase",
        choices=("all", "exp1", "exp2"),
        default="all",
        help="运行阶段",
    )
    args = p.parse_args()

    if not args.full and args.n is None and args.n_cmedqa is None and args.n_miriad is None:
        args.n = 25

    per_dataset_n: dict[str, int] = {}
    for dataset, _lang, _subset in DATASETS:
        override = args.n_cmedqa if dataset == "cmedqa" else args.n_miriad if dataset == "miriad" else None
        base = override if override is not None else args.n
        per_dataset_n[dataset] = resolve_sample_size(dataset, base, full=args.full)

    logger.info(f"sample sizes: {per_dataset_n}")

    paths: list[str] = []
    for dataset, language, subset in DATASETS:
        n = per_dataset_n[dataset]
        if args.phase in ("all", "exp1"):
            paths.append(
                _run_one(
                    phase="exp1",
                    dataset=dataset,
                    language=language,
                    subset=subset,
                    n=n,
                    conditions=_exp1_conditions(),
                )
            )
        if args.phase in ("all", "exp2"):
            paths.append(
                _run_one(
                    phase="exp2",
                    dataset=dataset,
                    language=language,
                    subset=subset,
                    n=n,
                    conditions=_exp2_conditions(),
                )
            )

    print("saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
