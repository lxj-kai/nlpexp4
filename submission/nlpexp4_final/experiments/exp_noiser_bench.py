"""NoiserBench 七类噪音实验：exp1 梯度 + exp2 矫正对比 + 可视化。

矩阵:
- exp1: naive × 7 noise types × ratios {0, 0.25, 0.5, 0.75, 1.0}
- exp2: methods × 7 noise types @ r=0.75 + clean baseline per method

用法:
    python -m experiments.exp_noiser_bench --n 50 --phase all
    python -m experiments.exp_noiser_bench --n 50 --phase exp2 --subset hotpotqa
    python -m experiments.exp_noiser_bench --n 2 --dry --phase exp1
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.config import CONFIG
from src.evaluator import Evaluator
from src.llm_client import LLMClient, get_judge_client
from src.noiser_loader import NOISER_NOISE_TYPES
from src.smoke_test import _DryLLM
from src.utils import get_logger, set_seed

logger = get_logger("exp_noiser_bench")

EXP1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXP2_METHODS = ("naive", "prompt", "confidence", "voting")
EXP2_RATIO = 0.75
DEFAULT_SUBSET = "hotpotqa"


def build_exp1_conditions() -> list[RunCondition]:
    return [
        RunCondition(
            method="naive",
            noise_ratio=ratio,
            noise_type=ntype,  # type: ignore[arg-type]
            noise_position="interleave",
            label=f"naive|r={ratio}|{ntype}",
        )
        for ratio, ntype in product(EXP1_RATIOS, NOISER_NOISE_TYPES)
    ]


def build_exp2_conditions(
    methods: tuple[str, ...] = EXP2_METHODS,
    *,
    ratio: float = EXP2_RATIO,
) -> list[RunCondition]:
    conds: list[RunCondition] = []
    for method in methods:
        conds.append(
            RunCondition(
                method=method,
                noise_ratio=0.0,
                noise_type="semantic",
                noise_position="interleave",
                label=f"{method}|clean",
            )
        )
    for method, ntype in product(methods, NOISER_NOISE_TYPES):
        conds.append(
            RunCondition(
                method=method,
                noise_ratio=ratio,
                noise_type=ntype,  # type: ignore[arg-type]
                noise_position="interleave",
                label=f"{method}|r={ratio}|{ntype}",
            )
        )
    return conds


def _run_phase(
    *,
    phase: str,
    subset: str,
    n: int,
    conditions: list[RunCondition],
    dry: bool,
    workers: int,
    render_figures: bool,
    figures_dir: Path | str | None,
    methods: tuple[str, ...] | None = None,
    ratio: float | None = None,
) -> dict:
    set_seed(CONFIG.seed)
    CONFIG.ensure_dirs()

    records = load_corpus(
        language="en",
        subset=subset,  # type: ignore[arg-type]
        dataset="noiser_bench",
        limit=n,
    )
    logger.info(
        f"noiser {phase} {subset}: {len(records)} samples × {len(conditions)} conditions"
    )

    llm: LLMClient | _DryLLM = _DryLLM() if dry else LLMClient()
    judge = None if dry else get_judge_client()
    evaluator = Evaluator(
        use_llm_judge=not dry,
        use_legacy_metrics=False,
        use_semantic_attribution=not dry,
        llm=llm,  # type: ignore[arg-type]
        judge_llm=judge if not dry else llm,  # type: ignore[arg-type]
    )

    results = run_conditions(
        records=records,
        conditions=conditions,
        llm=llm,  # type: ignore[arg-type]
        evaluator=evaluator,
        language="en",
        dataset="noiser_bench",
        show_progress=True,
        workers=workers,
    )

    exp_name = f"exp_noiser_{phase}_{subset}_n{n}"
    out_path = save_run(
        experiment_name=exp_name,
        results=results,
        extras={
            "phase": phase,
            "dataset": "noiser_bench",
            "subset": subset,
            "n_samples": n,
            "dry": dry,
            "noise_types": list(NOISER_NOISE_TYPES),
            "methods": list(methods or []),
            "ratio": ratio,
            "llm_usage": llm.usage.to_dict(),
        },
    )

    figure_paths: list[str] = []
    if render_figures and not dry:
        from src.visualize import render_noiser_bench_figures

        figure_paths = render_noiser_bench_figures(
            out_path,
            phase=phase,
            out_dir=figures_dir,
            tag=Path(out_path).stem,
        )

    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {
        "result_json": out_path,
        "figures": figure_paths,
        "robustness_table": payload.get("robustness_table", []),
        "phase": phase,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="NoiserBench 七类噪音实验")
    p.add_argument("--n", type=int, default=50, help="样本数")
    p.add_argument("--subset", default=DEFAULT_SUBSET, help="NoiserBench 子集（默认 hotpotqa）")
    p.add_argument(
        "--phase",
        choices=("all", "exp1", "exp2"),
        default="all",
        help="exp1=naive 梯度; exp2=矫正×七类噪音",
    )
    p.add_argument(
        "--methods",
        default=",".join(EXP2_METHODS),
        help="exp2 方法列表（逗号分隔）",
    )
    p.add_argument("--ratio", type=float, default=EXP2_RATIO, help="exp2 固定噪音比例")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--figures-dir", default=None)
    args = p.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    outputs: list[dict] = []

    if args.phase in ("all", "exp1"):
        outputs.append(
            _run_phase(
                phase="exp1",
                subset=args.subset,
                n=args.n,
                conditions=build_exp1_conditions(),
                dry=args.dry,
                workers=args.workers,
                render_figures=not args.no_figures,
                figures_dir=args.figures_dir,
            )
        )

    if args.phase in ("all", "exp2"):
        outputs.append(
            _run_phase(
                phase="exp2",
                subset=args.subset,
                n=args.n,
                conditions=build_exp2_conditions(methods, ratio=args.ratio),
                dry=args.dry,
                workers=args.workers,
                render_figures=not args.no_figures,
                figures_dir=args.figures_dir,
                methods=methods,
                ratio=args.ratio,
            )
        )

    print("saved:")
    for out in outputs:
        print(f"  [{out['phase']}] {out['result_json']}")
        for fig in out.get("figures", []):
            print(f"    figure -> {fig}")


if __name__ == "__main__":
    main()
