"""Build a 10-row MobileMem graph preview filtered by clean LLM behavior.

The preview set is for manual QA inspection. It keeps only samples where the
model answers correctly with the four positive documents and no noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import RGBRecord  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.noise_injector import NoisyContext, inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402


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


def as_record(row: dict[str, Any]) -> RGBRecord:
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
        subset="mobilemem_graph_hard_preview",
    )


# At noise_ratio=0 the only document orderings the UI can produce are
# "interleave" (positives sampled then shuffled again) and "front"/"back"/
# "surround" (positives in sampled order — identical prompt). Checking BOTH is
# required: a sample can be arithmetic-correct in one ordering and wrong in the
# other, which would let the UI show a wrong answer at zero noise.
_CLEAN_POSITIONS = ("interleave", "front")


def clean_context(record: RGBRecord, position: str = "interleave") -> NoisyContext:
    # Must mirror the backend graph-subset pipeline at noise_ratio=0 (same
    # seeded positive ordering, same max_docs) so that "keep" guarantees the UI
    # shows a correct answer with no noise.
    return inject(
        record,
        0.0,
        noise_type="semantic",
        noise_position=position,
        max_docs=80,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--start-id", type=int, default=340000)
    parser.add_argument(
        "--allow-operation",
        action="append",
        default=[],
        help="If set, keep only these operation types. Repeatable.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    llm = LLMClient()
    pipe = RAGPipeline(llm=llm)
    evaluator = Evaluator(use_llm_judge=False, llm=llm)

    allowed = set(args.allow_operation)
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        if len(kept) >= args.target:
            break
        meta = row.get("mobilemem_meta") or {}
        operation = str(meta.get("operation") or "")
        if allowed and operation not in allowed:
            continue
        # Evaluate with the SAME id the sample will have after renumbering: the
        # injector seed (and thus positive ordering) is derived from record.id,
        # so checking clean correctness under the original id would test a
        # different document order than the backend actually serves.
        final_id = args.start_id + len(kept)
        record = as_record(row)
        record.id = final_id
        preds: dict[str, str] = {}
        contains: dict[str, float] = {}
        ok = True
        for position in _CLEAN_POSITIONS:
            result = pipe.answer(clean_context(record, position), language="zh")
            metrics = evaluator.evaluate_one(result)
            preds[position] = result.prediction
            contains[position] = metrics.contains
            if metrics.contains < 1.0:
                ok = False
        audit.append(
            {
                "id": final_id,
                "source_id": row.get("id"),
                "decision": "keep" if ok else "drop_clean_wrong",
                "operation": operation,
                "metric": meta.get("metric"),
                "gold": record.answers_norm,
                "predictions": preds,
                "contains": contains,
            }
        )
        if not ok:
            continue
        out = deepcopy(row)
        out["id"] = final_id
        out_meta = out.setdefault("mobilemem_meta", {})
        out_meta["source_sample_id"] = row.get("id")
        out_meta["preview_filter"] = {
            "rule": "clean context must be correct in ALL UI orderings",
            "positions_checked": list(_CLEAN_POSITIONS),
            "clean_predictions": preds,
            "clean_contains": contains,
        }
        kept.append(out)

    write_jsonl(args.output, kept)
    audit_path = args.output.with_suffix(".clean_audit.jsonl")
    write_jsonl(audit_path, audit)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": str(audit_path),
                "seen": len(audit),
                "kept": len(kept),
                "target": args.target,
                "usage": llm.usage.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
