"""Verify the graph-hard preview is answerable with NO noise (read-only).

For every sample in zh_mobilemem_graph_hard_preview, rebuild the exact context
the backend serves at noise_ratio=0 (same kwargs as backend.routes.experiment),
run the naive pipeline, and check `contains`. This is the guarantee the user
relies on: "无噪音必须答对".

At noise=0 the only document orderings the UI can produce are:
  - interleave  -> positives sampled then shuffled again
  - front/back/surround -> positives in sampled order (identical prompt)
so testing {interleave, front} fully covers every UI position at noise=0.

Nothing is written; this only reads data + the LLM disk cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.deps import evaluator, find_record, get_records, llm  # noqa: E402
from backend.models import InjectRequest  # noqa: E402
from backend.routes.experiment import _inject_kwargs  # noqa: E402
from src.noise_injector import inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

SUBSET = "mobilemem_graph_hard_preview"
LANG = "zh"
POSITIONS = ["interleave", "front"]  # front == back == surround at noise=0


def main() -> None:
    rag = RAGPipeline(llm=llm)
    records = get_records(LANG, SUBSET)
    ids = sorted(r.id for r in records)
    print(f"# verify clean (noise=0) · subset={SUBSET} · {len(ids)} samples\n")

    all_ok = True
    fails: list[str] = []
    for sid in ids:
        record = find_record(LANG, SUBSET, sid)
        gold = " / ".join(record.answers_norm)
        meta = record.meta or {}
        line_head = (
            f"[{sid}] op={meta.get('operation')} metric={meta.get('metric')} "
            f"gold={gold}"
        )
        print(line_head)
        for pos in POSITIONS:
            req = InjectRequest(
                language=LANG,
                subset=SUBSET,
                sample_id=sid,
                noise_ratio=0.0,
                noise_type="semantic",
                noise_position=pos,
            )
            ctx = inject(record, **_inject_kwargs(req, record))
            result = rag.answer(ctx, language=LANG)
            metrics = evaluator.evaluate_one(result)
            ok = metrics.contains >= 1.0
            all_ok = all_ok and ok
            cached = result.metadata.get("cached", False)
            flag = "OK " if ok else "XX "
            if not ok:
                fails.append(f"{sid}/{pos}: pred={result.prediction!r} gold={gold}")
            print(
                f"    {flag} pos={pos:<10} docs={len(ctx.docs)} "
                f"contains={metrics.contains:.0f} cached={cached} "
                f"pred={result.prediction!r}"
            )
        print()

    print("=" * 60)
    print(f"usage this run: {llm.usage.to_dict()}")
    if all_ok:
        print(f"RESULT: ALL CLEAN-CORRECT ✓  ({len(ids)}/{len(ids)} samples, all UI positions)")
    else:
        print(f"RESULT: {len(fails)} FAILURE(S) ✗")
        for f in fails:
            print("  - " + f)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
