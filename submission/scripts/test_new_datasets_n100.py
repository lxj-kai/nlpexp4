"""Sanity check: naive RAG (clean) × 100 samples on bright / multihop_rag / tempo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._runner import RunCondition, load_corpus, run_conditions, save_run
from src.config import CONFIG
from src.evaluator import Evaluator, aggregate
from src.llm_client import LLMClient, get_judge_client
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

DATASETS = (
    ("bright", "en", "main"),
    ("multihop_rag", "en", "main"),
    ("tempo", "en", "main"),
)


def _sample_rows(rows: list[dict], n: int = 3) -> list[dict]:
    out = []
    for row in rows[:n]:
        out.append(
            {
                "id": row.get("sample_id"),
                "query": (row.get("query") or "")[:120],
                "gold": row.get("gold"),
                "pred": (row.get("prediction") or "")[:200],
                "judge_score": row.get("judge_score"),
                "judge_correct": row.get("judge_correct"),
            }
        )
    return out


def main() -> None:
    set_seed(CONFIG.seed)
    CONFIG.ensure_dirs()

    llm = LLMClient()
    judge = get_judge_client()
    logger.info(f"generation model: {CONFIG.model}")
    logger.info(f"judge model: {CONFIG.judge_model}")

    cond = RunCondition(
        method="naive",
        noise_ratio=0.0,
        noise_type="semantic",
        noise_position="interleave",
        label="naive|clean",
    )
    evaluator = Evaluator(
        use_llm_judge=True,
        use_legacy_metrics=False,
        use_semantic_attribution=False,
        llm=llm,
        judge_llm=judge,
    )

    report: dict = {"model": CONFIG.model, "datasets": {}}
    for dataset, language, subset in DATASETS:
        logger.info("=" * 60)
        logger.info(f"testing {dataset} n=100 ({language}/{subset})")
        records = load_corpus(
            language=language,  # type: ignore[arg-type]
            subset=subset,  # type: ignore[arg-type]
            dataset=dataset,  # type: ignore[arg-type]
            limit=100,
        )
        results = run_conditions(
            records=records,
            conditions=[cond],
            llm=llm,
            evaluator=evaluator,
            language=language,
            dataset=dataset,
            show_progress=True,
        )
        path = save_run(
            experiment_name=f"sanity_{dataset}_n100",
            results=results,
            extras={"dataset": dataset, "n": 100, "model": CONFIG.model},
        )
        rows = results[0].rows
        summary = aggregate(rows, group_by=("method",))[0]
        empty = sum(1 for r in rows if not (r.get("prediction") or "").strip())
        report["datasets"][dataset] = {
            "result_path": str(path),
            "n": len(rows),
            "empty_predictions": empty,
            "summary": summary,
            "samples": _sample_rows(rows),
        }
        logger.info(
            f"{dataset}: judge_score={summary.get('judge_score')} "
            f"judge_correct={summary.get('judge_correct')} empty={empty}"
        )

    out_path = CONFIG.results_dir / "sanity_new_datasets_n100_summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSummary -> {out_path}")


if __name__ == "__main__":
    main()
