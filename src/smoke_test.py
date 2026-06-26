"""批量评测脚本：多噪音比例 × 多矫正方法 → JSON 结果 + 可视化图表。

用法：
    python -m src.smoke_test --n 10 --dataset mobilemem --subset calc
    python -m src.smoke_test --n 5 --dry --methods naive,prompt,confidence
    python -m src.smoke_test --n 20 --methods naive,prompt,confidence,voting,iterative_sc --no-figures
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from .config import CONFIG
from .correctors import list_correctors
from .evaluator import Evaluator
from .llm_client import LLMClient, get_judge_client
from .utils import get_logger, set_seed

logger = get_logger(__name__)

DEFAULT_METHODS = ("naive", "prompt", "confidence", "voting")
DEFAULT_RATIOS = (0.0, 0.5, 0.75)


class _DryLLM:
    """假 LLM：离线校验 pipeline 接线。"""

    class _Usage:
        prompt_tokens = 0
        completion_tokens = 0
        calls = 0

        def add(self, *_):
            pass

        def to_dict(self):
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def __init__(self) -> None:
        self.usage = self._Usage()

    def chat(self, messages, **kwargs):
        user = messages[-1]["content"] if messages else ""
        return {
            "content": f"[DRY] echo: {user[:30]}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency": 0.0,
            "cached": True,
        }

    def generate(self, system, user, **kwargs):
        return self.chat([{"role": "user", "content": user}])["content"]


def build_conditions(
    methods: tuple[str, ...],
    ratios: tuple[float, ...],
    *,
    noise_type: str = "semantic",
    noise_position: str = "interleave",
) -> list[RunCondition]:
    return [
        RunCondition(
            method=m,
            noise_ratio=r,
            noise_type=noise_type,  # type: ignore[arg-type]
            noise_position=noise_position,  # type: ignore[arg-type]
            label=f"{m}|r={r}|t={noise_type}",
        )
        for m, r in product(methods, ratios)
    ]


def run(
    n: int = 5,
    *,
    dry: bool = False,
    language: str = "zh",
    subset: str = "main",
    dataset: str | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    noise_type: str = "semantic",
    noise_position: str = "interleave",
    workers: int = 1,
    render_figures: bool = True,
    figures_dir: Path | str | None = None,
    experiment_prefix: str = "smoke",
) -> dict:
    set_seed(CONFIG.seed)
    CONFIG.ensure_dirs()

    ds = dataset or CONFIG.dataset
    conditions = build_conditions(
        methods,
        ratios,
        noise_type=noise_type,
        noise_position=noise_position,
    )
    records = load_corpus(
        language=language,  # type: ignore[arg-type]
        subset=subset,  # type: ignore[arg-type]
        dataset=ds,
        limit=n,
    )
    logger.info(
        f"batch eval: {ds}/{language}/{subset} n={len(records)} "
        f"methods={methods} ratios={ratios} conditions={len(conditions)}"
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
        language=language,
        dataset=ds,
        show_progress=True,
        workers=workers,
    )

    exp_name = f"{experiment_prefix}_{ds}_{language}_{subset}_n{n}"
    out_path = save_run(
        experiment_name=exp_name,
        results=results,
        extras={
            "smoke": True,
            "dry": dry,
            "n_samples": n,
            "dataset": ds,
            "language": language,
            "subset": subset,
            "methods": list(methods),
            "ratios": list(ratios),
            "noise_type": noise_type,
            "noise_position": noise_position,
            "llm_usage": llm.usage.to_dict(),
        },
    )

    figure_paths: list[str] = []
    if render_figures and not dry:
        from .visualize import render_batch_run_figures

        figure_paths = render_batch_run_figures(
            out_path,
            out_dir=figures_dir,
            tag=Path(out_path).stem,
        )
    elif render_figures and dry:
        logger.info("skip figure render in --dry mode")

    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return {
        "result_json": out_path,
        "figures": figure_paths,
        "robustness_table": payload.get("robustness_table", []),
        "method_diagnostics": payload.get("method_diagnostics", {}),
        "n_samples": n,
        "dry": dry,
        "methods": list(methods),
        "ratios": list(ratios),
    }


def main() -> None:
    available = ["naive", *list_correctors()]
    p = argparse.ArgumentParser(description="批量评测：噪音梯度 × 矫正方法 + 可视化")
    p.add_argument("--n", type=int, default=5, help="样本数")
    p.add_argument("--dry", action="store_true", help="不真实调用 LLM")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument(
        "--subset",
        default="main",
        help="子集名（随数据集变化，如 main/fact 或 calc/noncalc）",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="默认读 NLP4_DATASET；如 rgb, mobilemem, bright, cmedqa, 2wiki, noiser_bench",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="方法列表（逗号分隔），可选: " + ",".join(available),
    )
    p.add_argument(
        "--ratios",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RATIOS),
        help="噪音比例（逗号分隔），如 0,0.5,0.75",
    )
    p.add_argument("--noise-type", default="semantic", help="semantic/counterfactual/mixed 等")
    p.add_argument(
        "--noise-position",
        default="interleave",
        choices=("front", "back", "interleave", "surround"),
    )
    p.add_argument("--workers", type=int, default=1, help="并行 worker 数")
    p.add_argument("--no-figures", action="store_true", help="跳过可视化出图")
    p.add_argument("--figures-dir", default=None, help="图表输出目录（默认 figures/<result_stem>/）")
    args = p.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in available]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; available: {available}")

    ratios = tuple(float(r) for r in args.ratios.split(",") if r.strip())
    result = run(
        n=args.n,
        dry=args.dry,
        language=args.language,
        subset=args.subset,
        dataset=args.dataset,
        methods=methods,
        ratios=ratios,
        noise_type=args.noise_type,
        noise_position=args.noise_position,
        workers=args.workers,
        render_figures=not args.no_figures,
        figures_dir=args.figures_dir,
    )

    print(json.dumps(result["robustness_table"], ensure_ascii=False, indent=2))
    print(f"\nresult -> {result['result_json']}")
    if result["figures"]:
        print("figures:")
        for fp in result["figures"]:
            print(f"  {fp}")


if __name__ == "__main__":
    main()
