"""Borrow cross-event noise for MobileMem-Reasoning.

This is stricter than ``borrow_mobilemem_noise.py``. Besides answer string
leakage, it rejects borrowed documents that can derive the same answer under
the target sample's reasoning rule.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.borrow_mobilemem_noise import (  # noqa: E402
    accept_doc,
    answers,
    event_key,
    grams,
    neighbor_order,
    norm,
    read_jsonl,
    source_candidates,
    write_jsonl,
)
from scripts.build_mobilemem_reasoning_dataset import doc_derives_answer  # noqa: E402


def derives_target_answer(doc: str, target: dict[str, Any]) -> bool:
    meta = target.get("mobilemem_meta") or {}
    answer_values = answers(target)
    if not answer_values:
        return False
    required = ("reasoning_type", "answer_field", "answer_artifact_type")
    if not all(meta.get(k) for k in required):
        return False
    return any(
        doc_derives_answer(
            doc,
            answer=answer,
            reasoning_type=str(meta["reasoning_type"]),
            answer_field=str(meta["answer_field"]),
            artifact_type=str(meta["answer_artifact_type"]),
        )
        for answer in answer_values
    )


def augment_row(
    rows: list[dict[str, Any]],
    i: int,
    *,
    target_negatives: int,
    neighbor_window: int,
    include_source_positive: bool,
    global_fill: bool,
    similarity_threshold: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    target = rows[i]
    out = deepcopy(target)
    seen: set[str] = set()
    kept_docs: list[str] = []
    kept_meta: list[dict[str, Any]] = []
    stats: dict[str, int] = {}

    def bump(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    positive_grams = [grams(d) for d in target.get("positive") or []]
    answer_norms = [norm(a) for a in answers(target)]

    for doc in target.get("negative") or []:
        ok, reason = accept_doc(
            doc=str(doc),
            target=target,
            source=None,
            seen=seen,
            positive_grams=positive_grams,
            answer_norms=answer_norms,
            similarity_threshold=similarity_threshold,
        )
        if not ok:
            bump(f"original_{reason}")
            continue
        if derives_target_answer(str(doc), target):
            bump("original_derives_answer")
            continue
        seen.add(norm(doc))
        kept_docs.append(str(doc))
        kept_meta.append(
            {
                "source": "original_negative",
                "source_sample_id": target.get("id"),
                "source_event_id": None,
                "source_event_name": None,
                "note": "existing negative from the target sample before cross-event borrowing",
            }
        )
        if len(kept_docs) >= target_negatives:
            break

    for j in neighbor_order(i, len(rows), neighbor_window, global_fill=global_fill):
        if len(kept_docs) >= target_negatives:
            break
        source = rows[j]
        if event_key(source) == event_key(target):
            continue
        for doc, role in source_candidates(source, include_source_positive=include_source_positive):
            if len(kept_docs) >= target_negatives:
                break
            ok, reason = accept_doc(
                doc=doc,
                target=target,
                source=source,
                seen=seen,
                positive_grams=positive_grams,
                answer_norms=answer_norms,
                similarity_threshold=similarity_threshold,
            )
            if not ok:
                bump(f"borrowed_{reason}")
                continue
            if derives_target_answer(doc, target):
                bump("borrowed_derives_answer")
                continue
            seen.add(norm(doc))
            kept_docs.append(doc)
            kept_meta.append(
                {
                    "source": role,
                    "source_sample_id": source.get("id"),
                    "source_event_id": (source.get("mobilemem_meta") or {}).get("event_id"),
                    "source_event_name": (source.get("mobilemem_meta") or {}).get("event_name"),
                    "distance": abs(j - i),
                }
            )

    out["negative"] = kept_docs
    out["negative_meta"] = kept_meta
    meta = out.setdefault("mobilemem_meta", {})
    meta["noise_policy"] = "semantic_cross_event_non_contradictory_non_supporting_reasoning"
    meta["target_negative_docs"] = target_negatives
    meta["actual_negative_docs"] = len(kept_docs)
    meta["borrowed_negative_docs"] = sum(1 for m in kept_meta if str(m.get("source", "")).startswith("borrowed"))
    meta["original_negative_docs"] = sum(1 for m in kept_meta if m.get("source") == "original_negative")
    return out, stats


def augment_dataset(
    rows: list[dict[str, Any]],
    *,
    target_negatives: int,
    neighbor_window: int,
    include_source_positive: bool,
    global_fill: bool,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    all_stats: dict[str, int] = {}
    for i in range(len(rows)):
        row, stats = augment_row(
            rows,
            i,
            target_negatives=target_negatives,
            neighbor_window=neighbor_window,
            include_source_positive=include_source_positive,
            global_fill=global_fill,
            similarity_threshold=similarity_threshold,
        )
        out.append(row)
        for k, v in stats.items():
            all_stats[k] = all_stats.get(k, 0) + v

    counts = [len(r.get("negative") or []) for r in out]
    summary = {
        "rows": len(out),
        "target_negatives": target_negatives,
        "min_negatives": min(counts) if counts else 0,
        "max_negatives": max(counts) if counts else 0,
        "avg_negatives": round(statistics.mean(counts), 3) if counts else 0,
        "underfilled": sum(c < target_negatives for c in counts),
        "filter_stats": all_stats,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/rgb/zh_mobilemem_reasoning.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/rgb/zh_mobilemem_reasoning_cross.json")
    parser.add_argument("--target-negatives", type=int, default=20)
    parser.add_argument("--neighbor-window", type=int, default=8)
    parser.add_argument(
        "--no-source-positive",
        action="store_true",
        help="Borrow only other samples' negative docs, not their positive docs.",
    )
    parser.add_argument(
        "--no-global-fill",
        action="store_true",
        help="Only borrow within the neighbor window; do not scan the rest if underfilled.",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out, summary = augment_dataset(
        rows,
        target_negatives=args.target_negatives,
        neighbor_window=args.neighbor_window,
        include_source_positive=not args.no_source_positive,
        global_fill=not args.no_global_fill,
        similarity_threshold=args.similarity_threshold,
    )
    write_jsonl(args.output, out)
    print(json.dumps({"output": str(args.output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
