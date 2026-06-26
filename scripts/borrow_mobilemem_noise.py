"""Borrow cross-event semantic noise for MobileMem QA data.

The input data already has clean positives and a small negative pool. This
script expands each sample's negative pool by borrowing documents from nearby
samples, while enforcing the MobileMem rule:

    noise may be related, but it must not support or contradict the answer.

The output keeps RGB-compatible fields and adds optional ``negative_meta`` for
auditing where each negative document came from.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PUNCT_RE = re.compile(r"[\s，。、；：？！,\.;:?!\"'`“”‘’（）()\[\]【】《》<>—\-]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: object) -> str:
    return PUNCT_RE.sub("", str(text).lower())


def answers(row: dict[str, Any]) -> list[str]:
    raw = row.get("answer", [])
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x)]
    return [str(raw)] if raw else []


def event_key(row: dict[str, Any]) -> tuple[object, object]:
    meta = row.get("mobilemem_meta") or {}
    return meta.get("uid"), meta.get("event_id")


def event_name(row: dict[str, Any]) -> str:
    return str((row.get("mobilemem_meta") or {}).get("event_name") or "")


def grams(text: str, n: int = 5) -> set[str]:
    text = norm(text)
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def neighbor_order(i: int, n: int, window: int, *, global_fill: bool) -> list[int]:
    ordered: list[int] = []
    seen = {i}
    for step in range(1, window + 1):
        for j in (i + step, i - step):
            if 0 <= j < n and j not in seen:
                ordered.append(j)
                seen.add(j)
    if global_fill:
        for step in range(window + 1, n):
            for j in (i + step, i - step):
                if 0 <= j < n and j not in seen:
                    ordered.append(j)
                    seen.add(j)
    return ordered


def accept_doc(
    *,
    doc: str,
    target: dict[str, Any],
    source: dict[str, Any] | None,
    seen: set[str],
    positive_grams: list[set[str]],
    answer_norms: list[str],
    similarity_threshold: float,
) -> tuple[bool, str]:
    doc_norm = norm(doc)
    if not doc_norm:
        return False, "empty"
    if doc_norm in seen:
        return False, "duplicate"

    if source is not None and event_key(source) == event_key(target):
        return False, "same_event_id"

    target_event_name = norm(event_name(target))
    if len(target_event_name) >= 6 and target_event_name in doc_norm:
        return False, "same_event_name"

    for ans in answer_norms:
        if ans and ans in doc_norm:
            return False, "contains_answer"

    doc_grams = grams(doc)
    if any(jaccard(doc_grams, pg) >= similarity_threshold for pg in positive_grams):
        return False, "too_similar_to_positive"

    return True, "accepted"


def source_candidates(
    source: dict[str, Any],
    *,
    include_source_positive: bool,
) -> Iterable[tuple[str, str]]:
    for doc in source.get("negative") or []:
        yield str(doc), "borrowed_negative"
    if include_source_positive:
        for doc in source.get("positive") or []:
            yield str(doc), "borrowed_positive"


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

    # Keep existing negatives first if they are still valid for the target.
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

    # Borrow from neighboring samples, then optionally from the rest of the file.
    for j in neighbor_order(i, len(rows), neighbor_window, global_fill=global_fill):
        if len(kept_docs) >= target_negatives:
            break
        source = rows[j]
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
    meta["noise_policy"] = "semantic_cross_event_non_contradictory"
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/rgb/zh_mobilemem.json")
    p.add_argument("--output", type=Path, default=ROOT / "data/rgb/zh_mobilemem_cross.json")
    p.add_argument("--target-negatives", type=int, default=20)
    p.add_argument("--neighbor-window", type=int, default=8)
    p.add_argument(
        "--no-source-positive",
        action="store_true",
        help="Borrow only other samples' negative docs, not their positive docs.",
    )
    p.add_argument(
        "--no-global-fill",
        action="store_true",
        help="Only borrow within the neighbor window; do not scan the rest if underfilled.",
    )
    p.add_argument("--similarity-threshold", type=float, default=0.82)
    args = p.parse_args()

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
