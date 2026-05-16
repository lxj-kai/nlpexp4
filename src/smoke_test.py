"""端到端 smoke test：跑 5 条样本 × 2 噪音比例 × baseline + prompt 矫正。

用法：
    python -m src.smoke_test [--n 5] [--dry]

`--dry` 模式不会真实调用 LLM，只用占位答案，用于校验 pipeline 接线是否正确。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .config import CONFIG
from .correctors import get_corrector, list_correctors
from .data_loader import load_dataset
from .evaluator import Evaluator, aggregate
from .llm_client import LLMClient
from .noise_injector import batch_inject
from .rag_pipeline import RAGPipeline, RAGResult
from .utils import get_logger, now_tag, set_seed, write_json, Timer

logger = get_logger(__name__)


class _DryLLM:
    """假 LLM：返回 gold[0] 或固定占位，便于离线 smoke test。"""

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


def run(n: int = 5, *, dry: bool = False, methods: tuple[str, ...] = ("naive", "prompt")) -> dict:
    set_seed(CONFIG.seed)
    CONFIG.ensure_dirs()

    records = load_dataset("zh", "main", limit=n)
    logger.info(f"loaded {len(records)} samples from zh/main")

    contexts_clean = batch_inject(records, noise_ratio=0.0)
    contexts_noisy = batch_inject(records, noise_ratio=0.5, noise_type="semantic")

    llm = _DryLLM() if dry else LLMClient()
    evaluator = Evaluator(use_llm_judge=False)

    all_rows: list[dict] = []
    for label, ctxs in [("clean", contexts_clean), ("noisy", contexts_noisy)]:
        for method in methods:
            with Timer(f"{label}/{method}"):
                if method == "naive":
                    pipe = RAGPipeline(llm=llm)  # type: ignore[arg-type]
                    results = pipe.batch_answer(ctxs, show_progress=False)
                else:
                    corr = get_corrector(method, llm=llm)  # type: ignore[arg-type]
                    results = corr.batch_correct(ctxs, show_progress=False)
            rows = evaluator.evaluate_batch(results)
            for r in rows:
                r["bucket"] = label
            all_rows.extend(rows)

    summaries = aggregate(all_rows, group_by=("method", "bucket"))

    out = {
        "config": CONFIG.to_dict(),
        "n_samples": n,
        "dry": dry,
        "methods": list(methods),
        "summaries": summaries,
        "rows": all_rows,
        "llm_usage": llm.usage.to_dict(),
    }
    out_path = CONFIG.results_dir / f"smoke_{now_tag()}.json"
    write_json(out, out_path)
    logger.info(f"smoke results -> {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="样本数")
    p.add_argument("--dry", action="store_true", help="不真实调用 LLM")
    p.add_argument(
        "--methods",
        type=str,
        default="naive,prompt",
        help="方法列表（逗号分隔），可选: " + ",".join(["naive", *list_correctors()]),
    )
    args = p.parse_args()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    result = run(n=args.n, dry=args.dry, methods=methods)
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
