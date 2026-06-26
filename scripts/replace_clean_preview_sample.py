"""Replace a single preview sample with one that is clean-correct in EVERY
UI document ordering (interleave AND front/back/surround).

Keeps all other preview rows byte-identical (same id / content / warmed cache),
so only the flaky slot changes. Backs up the preview before writing.
"""
from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.models import InjectRequest  # noqa: E402
from backend.routes.experiment import _inject_kwargs  # noqa: E402
from src.data_loader import RGBRecord  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.noise_injector import inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

PREVIEW = ROOT / "data" / "rgb" / "zh_mobilemem_graph_hard_preview.json"
POOL = ROOT / "data" / "rgb" / "_cw_pool_tmp.json"
SUBSET = "mobilemem_graph_hard_preview"
LANG = "zh"
POSITIONS = ["interleave", "front"]  # front == back == surround at noise=0
REPLACE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 340006


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sig(row: dict[str, Any]) -> tuple:
    return tuple(sorted((row.get("mobilemem_meta") or {}).get("event_ids") or []))


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


def clean_ok_all_positions(rag, evaluator, record) -> tuple[bool, list[str]]:
    preds = []
    for pos in POSITIONS:
        req = InjectRequest(language=LANG, subset=SUBSET, sample_id=record.id,
                            noise_ratio=0.0, noise_type="semantic", noise_position=pos)
        ctx = inject(record, **_inject_kwargs(req, record))
        result = rag.answer(ctx, language=LANG)
        m = evaluator.evaluate_one(result)
        preds.append(f"{pos}={result.prediction!r}({m.contains:.0f})")
        if m.contains < 1.0:
            return False, preds
    return True, preds


def main() -> None:
    preview = read_jsonl(PREVIEW)
    pool = read_jsonl(POOL)
    by_id = {r["id"]: r for r in preview}
    if REPLACE_ID not in by_id:
        print(f"id {REPLACE_ID} not in preview"); sys.exit(1)

    keep_sigs = {sig(r) for rid, r in by_id.items() if rid != REPLACE_ID}
    bad_sig = sig(by_id[REPLACE_ID])

    llm = LLMClient()
    rag = RAGPipeline(llm=llm)
    evaluator = Evaluator(use_llm_judge=False, llm=llm)

    chosen = None
    chosen_preds: list[str] = []
    for cand in pool:
        s = sig(cand)
        if s in keep_sigs or s == bad_sig:
            continue
        rec = as_record(cand, REPLACE_ID)
        ok, preds = clean_ok_all_positions(rag, evaluator, rec)
        mark = "PICK" if ok else "skip"
        print(f"  [{mark}] src={cand['id']} gold={rec.answers_norm} :: {' | '.join(preds)}")
        if ok:
            chosen = cand
            chosen_preds = preds
            break

    if chosen is None:
        print("No clean-robust replacement found in pool."); sys.exit(1)

    new_row = deepcopy(chosen)
    new_row["id"] = REPLACE_ID
    meta = new_row.setdefault("mobilemem_meta", {})
    meta["source_sample_id"] = chosen["id"]
    meta["preview_filter"] = {
        "rule": "clean context must be correct in ALL UI orderings",
        "positions_checked": POSITIONS,
        "clean_predictions": chosen_preds,
    }

    new_preview = [new_row if r["id"] == REPLACE_ID else r for r in preview]

    backup = PREVIEW.with_suffix(".json.bak_before_robust")
    shutil.copy2(PREVIEW, backup)
    write_jsonl(PREVIEW, new_preview)
    print("\n" + "=" * 60)
    print(f"replaced id {REPLACE_ID}: src {chosen['id']} -> gold {as_record(chosen, REPLACE_ID).answers_norm}")
    print(f"backup: {backup}")
    print(f"usage: {llm.usage.to_dict()}")


if __name__ == "__main__":
    main()
