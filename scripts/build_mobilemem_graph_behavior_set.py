"""Generate MobileMem graph data under production-style behavioral constraints.

Target constraint:

- Naive RAG with 0% noise answers correctly.
- Naive RAG with 70% semantic noise answers incorrectly across one or more
  deterministic noise seeds.

Pipeline stages:

1. Generate graph-style candidate QA records from MobileMem.
2. Apply static structural checks before any LLM calls.
3. Verify clean-context Naive RAG correctness.
4. Verify 70%-noise Naive RAG failure over multiple noise seeds.
5. Enforce simple diversity quotas and write audit + manifest files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_graph_hard_dataset import (  # noqa: E402
    build_graph_hard_dataset,
)
from src.data_loader import RGBRecord  # noqa: E402
from src.evaluator import Evaluator, normalize_answer  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.noise_injector import NoisyContext, inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402


VARIANTS: dict[str, dict[str, Any]] = {
    "v1": {
        "allow_operations": {"sum", "range"},
        "allow_metrics": {
            "shopping_price",
            "money_amount",
            "book_minutes",
            "chat_sender_count",
            "video_like_gap",
            "video_like_danmaku_sum",
            "music_remaining_seconds",
        },
        "negative_docs": 30,
        "seed": 42,
    },
    "v2": {
        "allow_operations": {"sum"},
        "allow_metrics": {
            "shopping_price",
            "money_amount",
            "book_minutes",
            "chat_sender_count",
        },
        "negative_docs": 30,
        "seed": 43,
    },
    "v3": {
        "allow_operations": {"sum", "range"},
        "allow_metrics": {
            "video_like_gap",
            "video_like_danmaku_sum",
            "music_remaining_seconds",
            "shopping_price",
            "money_amount",
        },
        "negative_docs": 30,
        "seed": 44,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_answers(row: dict[str, Any]) -> list[str]:
    raw = row.get("answer") or []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x)]
    return [str(raw)] if raw else []


def as_record(row: dict[str, Any], *, subset: str) -> RGBRecord:
    answer = row.get("answer") or []
    if not isinstance(answer, list):
        answer = [str(answer)]
    return RGBRecord(
        id=int(row["id"]),
        query=str(row["query"]),
        answer=[str(x) for x in answer],
        positive=[str(x) for x in row.get("positive") or []],
        negative=[str(x) for x in row.get("negative") or []],
        positive_wrong=[],
        fakeanswer="",
        meta=dict(row.get("mobilemem_meta") or {}),
        language="zh",
        subset=subset,
    )


def clean_context(record: RGBRecord) -> NoisyContext:
    return NoisyContext(
        sample_id=record.id,
        query=record.query,
        gold_answers=record.answers_norm,
        docs=list(record.positive),
        labels=["positive"] * len(record.positive),
        noise_ratio=0.0,
        noise_type="semantic",
        noise_position="interleave",
        meta={"total": len(record.positive), "positives": len(record.positive), "noises": 0},
    )


def noisy70_context(record: RGBRecord, *, seed: int) -> NoisyContext:
    return inject(
        record,
        0.70,
        noise_type="semantic",
        noise_position="interleave",
        max_docs=80,
        min_positive=min(4, len(record.positive)),
        keep_all_positive=True,
        seed=seed,
    )


def static_reject_reasons(row: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    meta = row.get("mobilemem_meta") or {}
    operation = str(meta.get("operation") or "")
    metric = str(meta.get("metric") or "")
    if operation not in variant["allow_operations"]:
        reasons.append("operation_not_allowed")
    if metric not in variant["allow_metrics"]:
        reasons.append("metric_not_allowed")
    if len(row.get("positive") or []) != 4:
        reasons.append("positive_count_not_4")
    if len(row.get("negative") or []) < 10:
        reasons.append("not_enough_negative_docs")

    answers = row_answers(row)
    query_norm = normalize_answer(str(row.get("query") or ""))
    if any(answer and normalize_answer(answer) in query_norm for answer in answers):
        reasons.append("answer_leaks_in_query")
    for doc in row.get("negative") or []:
        doc_norm = normalize_answer(str(doc))
        if any(answer and normalize_answer(answer) in doc_norm for answer in answers):
            reasons.append("answer_leaks_in_negative")
            break

    pos_norms = [normalize_answer(str(doc)) for doc in row.get("positive") or []]
    if len(set(pos_norms)) != len(pos_norms):
        reasons.append("duplicate_positive_docs")

    if operation in {"sum", "range"}:
        for doc in row.get("positive") or []:
            doc_norm = normalize_answer(str(doc))
            if any(answer and normalize_answer(answer) in doc_norm for answer in answers):
                reasons.append("derived_answer_leaks_in_positive")
                break

    if meta.get("single_doc_answerable") is not False:
        reasons.append("single_doc_answerable_not_false")
    if meta.get("required_positive_docs") != 4:
        reasons.append("required_positive_docs_not_4")

    return reasons


def contains_gold_text(prediction: str, golds: list[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    return any(g and normalize_answer(g) in pred_norm for g in golds)


def metric_count(rows: list[dict[str, Any]], metric: str) -> int:
    return sum(1 for row in rows if (row.get("mobilemem_meta") or {}).get("metric") == metric)


def operation_count(rows: list[dict[str, Any]], operation: str) -> int:
    return sum(1 for row in rows if (row.get("mobilemem_meta") or {}).get("operation") == operation)


def quota_reject_reason(
    row: dict[str, Any],
    kept: list[dict[str, Any]],
    *,
    max_per_metric: int,
    max_per_operation: int,
) -> str | None:
    meta = row.get("mobilemem_meta") or {}
    metric = str(meta.get("metric") or "")
    operation = str(meta.get("operation") or "")
    if max_per_metric > 0 and metric_count(kept, metric) >= max_per_metric:
        return "metric_quota_full"
    if max_per_operation > 0 and operation_count(kept, operation) >= max_per_operation:
        return "operation_quota_full"
    return None


def evaluate_variant(
    rows: list[dict[str, Any]],
    *,
    variant_name: str,
    variant: dict[str, Any],
    target: int,
    start_id: int,
    llm: LLMClient,
    pipe: RAGPipeline,
    evaluator: Evaluator,
    noisy_trials: int,
    max_per_metric: int,
    max_per_operation: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for row in rows:
        if len(kept) >= target:
            break
        static_reasons = static_reject_reasons(row, variant)
        if static_reasons:
            audit.append(
                {
                    "variant": variant_name,
                    "id": row.get("id"),
                    "decision": "drop_static",
                    "reasons": static_reasons,
                    "operation": (row.get("mobilemem_meta") or {}).get("operation"),
                    "metric": (row.get("mobilemem_meta") or {}).get("metric"),
                }
            )
            continue
        quota_reason = quota_reject_reason(
            row,
            kept,
            max_per_metric=max_per_metric,
            max_per_operation=max_per_operation,
        )
        if quota_reason:
            audit.append(
                {
                    "variant": variant_name,
                    "id": row.get("id"),
                    "decision": "drop_quota",
                    "reason": quota_reason,
                    "operation": (row.get("mobilemem_meta") or {}).get("operation"),
                    "metric": (row.get("mobilemem_meta") or {}).get("metric"),
                }
            )
            continue

        record = as_record(row, subset="mobilemem_graph_hard_preview")
        clean_result = pipe.answer(clean_context(record), language="zh")
        clean_metrics = evaluator.evaluate_one(clean_result)
        if clean_metrics.contains < 1.0:
            audit.append(
                {
                    "variant": variant_name,
                    "id": row.get("id"),
                    "decision": "drop_clean_wrong",
                    "gold": record.answers_norm,
                    "clean_prediction": clean_result.prediction,
                    "clean_contains": clean_metrics.contains,
                    "operation": record.meta.get("operation"),
                    "metric": record.meta.get("metric"),
                }
            )
            continue

        noisy_trials_out: list[dict[str, Any]] = []
        noisy_failed_all_trials = True
        for trial in range(noisy_trials):
            noise_seed = int(variant["seed"]) + int(row["id"]) + trial * 1_000_003
            noisy_ctx = noisy70_context(record, seed=noise_seed)
            noisy_result = pipe.answer(noisy_ctx, language="zh")
            noisy_metrics = evaluator.evaluate_one(noisy_result)
            noisy_hit = noisy_metrics.contains >= 1.0 or contains_gold_text(
                noisy_result.prediction, record.answers_norm
            )
            noisy_trials_out.append(
                {
                    "trial": trial,
                    "seed": noise_seed,
                    "prediction": noisy_result.prediction,
                    "contains": noisy_metrics.contains,
                    "text_hit": noisy_hit,
                    "actual_noise_ratio": noisy_ctx.noise_ratio,
                    "docs": len(noisy_ctx.docs),
                    "positives": sum(label == "positive" for label in noisy_ctx.labels),
                    "noises": sum(label != "positive" for label in noisy_ctx.labels),
                }
            )
            if noisy_hit:
                noisy_failed_all_trials = False
                break

        if not noisy_failed_all_trials:
            audit.append(
                {
                    "variant": variant_name,
                    "id": row.get("id"),
                    "decision": "drop_noisy_correct",
                    "gold": record.answers_norm,
                    "clean_prediction": clean_result.prediction,
                    "noisy_trials": noisy_trials_out,
                    "operation": record.meta.get("operation"),
                    "metric": record.meta.get("metric"),
                }
            )
            continue

        out = deepcopy(row)
        out["id"] = start_id + len(kept)
        meta = out.setdefault("mobilemem_meta", {})
        meta["source_sample_id"] = row.get("id")
        meta["behavior_filter"] = {
            "variant": variant_name,
            "rule": "keep iff naive clean correct and naive 70pct-noise wrong",
            "clean_prediction": clean_result.prediction,
            "clean_contains": clean_metrics.contains,
            "noisy_trials": noisy_trials_out,
            "target_noise_ratio": 0.70,
            "noise_position": "interleave",
            "max_docs": 80,
            "min_positive": 4,
            "keep_all_positive": True,
            "noisy_trial_count": noisy_trials,
        }
        kept.append(out)
        audit.append(
            {
                "variant": variant_name,
                "id": row.get("id"),
                "new_id": out["id"],
                "decision": "keep",
                "gold": record.answers_norm,
                "clean_prediction": clean_result.prediction,
                "noisy_trials": noisy_trials_out,
                "operation": record.meta.get("operation"),
                "metric": record.meta.get("metric"),
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant_name,
                    "kept": len(kept),
                    "id": row.get("id"),
                    "gold": record.answers_norm,
                    "clean": clean_result.prediction,
                    "noisy_trials": [trial["prediction"] for trial in noisy_trials_out],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return kept, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT.parent / "data" / "0418(1)" / "data",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/rgb/zh_mobilemem_graph_hard_preview.json")
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--start-id", type=int, default=340000)
    parser.add_argument("--candidate-limit", type=int, default=160)
    parser.add_argument("--candidate-start-id", type=int, default=350000)
    parser.add_argument(
        "--noisy-trials",
        type=int,
        default=2,
        help="Number of deterministic 70 percent-noise samples that must all fail.",
    )
    parser.add_argument("--max-per-metric", type=int, default=4)
    parser.add_argument("--max-per-operation", type=int, default=6)
    args = parser.parse_args()

    tmp_dir = ROOT / "data/rgb/.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()
    pipe = RAGPipeline(llm=llm)
    evaluator = Evaluator(use_llm_judge=False, llm=llm)

    final_rows: list[dict[str, Any]] = []
    final_audit: list[dict[str, Any]] = []
    chosen_variant = None

    for variant_name, variant in VARIANTS.items():
        candidate_path = tmp_dir / f"mobilemem_graph_{variant_name}_candidates.json"
        build_graph_hard_dataset(
            input_path=args.input_root,
            output_path=candidate_path,
            language="zh",
            limit=args.candidate_limit,
            seed=int(variant["seed"]),
            start_id=args.candidate_start_id,
            negative_docs=int(variant["negative_docs"]),
        )
        rows = read_jsonl(candidate_path)
        kept, audit = evaluate_variant(
            rows,
            variant_name=variant_name,
            variant=variant,
            target=args.target,
            start_id=args.start_id,
            llm=llm,
            pipe=pipe,
            evaluator=evaluator,
            noisy_trials=args.noisy_trials,
            max_per_metric=args.max_per_metric,
            max_per_operation=args.max_per_operation,
        )
        final_audit.extend(audit)
        if len(kept) >= args.target:
            final_rows = kept[: args.target]
            chosen_variant = variant_name
            break

    if len(final_rows) < args.target:
        raise SystemExit(f"only kept {len(final_rows)}/{args.target}; inspect audit")

    write_jsonl(args.output, final_rows)
    audit_path = args.output.with_suffix(".behavior_audit.jsonl")
    write_jsonl(audit_path, final_audit)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "output": str(args.output),
        "audit": str(audit_path),
        "target": args.target,
        "kept": len(final_rows),
        "chosen_variant": chosen_variant,
        "constraints": {
            "clean": "naive RAG over 4 positive docs must contain gold",
            "noisy": "naive RAG over 4 positive docs plus 70% semantic noise must not contain gold",
            "noisy_trials": args.noisy_trials,
            "max_per_metric": args.max_per_metric,
            "max_per_operation": args.max_per_operation,
            "target_noise_ratio": 0.70,
            "keep_all_positive": True,
        },
        "variants": VARIANTS,
        "operations": dict(Counter((r.get("mobilemem_meta") or {}).get("operation") for r in final_rows)),
        "metrics": dict(Counter((r.get("mobilemem_meta") or {}).get("metric") for r in final_rows)),
        "drop_reasons": dict(Counter(a.get("decision") for a in final_audit)),
        "usage": llm.usage.to_dict(),
    }
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": str(audit_path),
                "manifest": str(manifest_path),
                "target": args.target,
                "kept": len(final_rows),
                "chosen_variant": chosen_variant,
                "operations": dict(Counter((r.get("mobilemem_meta") or {}).get("operation") for r in final_rows)),
                "metrics": dict(Counter((r.get("mobilemem_meta") or {}).get("metric") for r in final_rows)),
                "drop_reasons": manifest["drop_reasons"],
                "usage": llm.usage.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
