"""Filter the multihop set down to a clean-correct preview.

Keeps only samples the model answers correctly with NO noise (the chain's
positive docs only) in EVERY UI ordering, using the exact backend pipeline.
This guarantees "无噪音必对" and that the model can reproduce the entity name.
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

from backend.deps import evaluator, llm  # noqa: E402
from backend.models import InjectRequest  # noqa: E402
from backend.routes.experiment import _inject_kwargs  # noqa: E402
from src.data_loader import RGBRecord  # noqa: E402
from src.noise_injector import inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

SUBSET = "mobilemem_multihop_preview"
LANG = "zh"
POSITIONS = ["interleave", "front"]  # front == back == surround at noise=0


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def as_record(row: dict[str, Any], rid: int) -> RGBRecord:
    ans = row.get("answer") or []
    if not isinstance(ans, list):
        ans = [str(ans)]
    return RGBRecord(
        id=rid,
        query=str(row["query"]),
        answer=[str(x) for x in ans],
        positive=[str(x) for x in row.get("positive") or []],
        negative=[str(x) for x in row.get("negative") or []],
        positive_wrong=[],
        fakeanswer="",
        meta=dict(row.get("mobilemem_meta") or {}),
        language=LANG,
        subset=SUBSET,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "rgb" / "zh_mobilemem_multihop.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "rgb" / "zh_mobilemem_multihop_preview.json")
    parser.add_argument("--start-id", type=int, default=351000)
    parser.add_argument("--target", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rag = RAGPipeline(llm=llm)
    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row in rows:
        if len(kept) >= args.target:
            break
        final_id = args.start_id + len(kept)
        record = as_record(row, final_id)
        preds: dict[str, str] = {}
        contains: dict[str, float] = {}
        ok = True
        for pos in POSITIONS:
            req = InjectRequest(language=LANG, subset=SUBSET, sample_id=final_id,
                                noise_ratio=0.0, noise_type="semantic", noise_position=pos)
            ctx = inject(record, **_inject_kwargs(req, record))
            result = rag.answer(ctx, language=LANG)
            m = evaluator.evaluate_one(result)
            preds[pos] = result.prediction
            contains[pos] = m.contains
            if m.contains < 1.0:
                ok = False
        audit.append({
            "id": final_id, "source_id": row.get("id"),
            "decision": "keep" if ok else "drop_clean_wrong",
            "gold": record.answers_norm, "predictions": preds, "contains": contains,
        })
        if not ok:
            continue
        out = deepcopy(row)
        out["id"] = final_id
        meta = out.setdefault("mobilemem_meta", {})
        meta["source_sample_id"] = row.get("id")
        meta["preview_filter"] = {
            "rule": "clean correct in ALL UI orderings",
            "positions_checked": POSITIONS,
            "clean_predictions": preds,
        }
        kept.append(out)

    write_jsonl(args.output, kept)
    write_jsonl(args.output.with_suffix(".clean_audit.jsonl"), audit)
    print(json.dumps({
        "output": str(args.output), "seen": len(audit), "kept": len(kept),
        "usage": llm.usage.to_dict(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
