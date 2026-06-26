"""Filter MobileMem samples by clean/noisy LLM behavior.

Keeps samples that satisfy:
- clean context answers correctly;
- high-noise context answers incorrectly.

For MobileMem-Reasoning cross95, high-noise means 2 positive docs plus 38
negative docs, i.e. 95% noise while keeping both required evidence docs.
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

from src.config import CONFIG  # noqa: E402
from src.data_loader import RGBRecord, load_dataset  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.noise_injector import NoisyContext, inject  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402


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


def noisy_context(record: RGBRecord, *, seed: int | None) -> NoisyContext:
    # For cross95 data: 2 positives + 38 negatives = 40 docs, actual 95% noise.
    return inject(
        record,
        0.95,
        noise_type="semantic",
        noise_position="interleave",
        max_docs=40,
        min_positive=2,
        seed=seed,
    )


def read_source_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row["id"])] = row
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--subset", default="mobilemem_reasoning_cross95")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/rgb/zh_mobilemem_reasoning_cross95.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/rgb/zh_mobilemem_reasoning_challenging.json",
    )
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--start-id", type=int, default=330000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-reasoning-type",
        action="append",
        default=[],
        help="Skip samples with this mobilemem_meta.reasoning_type. Can be repeated.",
    )
    args = parser.parse_args()

    if not CONFIG.api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")

    source_rows = read_source_rows(args.input)
    all_records = load_dataset(args.language, args.subset, limit=None, shuffle=False)
    records = all_records[args.offset : args.offset + args.limit]
    llm = LLMClient()
    pipe = RAGPipeline(llm=llm)
    evaluator = Evaluator(use_llm_judge=False, llm=llm)

    kept: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    audit_path = args.output.with_suffix(".audit.jsonl")
    excluded_reasoning_types = set(args.exclude_reasoning_type)

    def checkpoint() -> None:
        write_jsonl(args.output, kept)
        write_jsonl(audit_path, audit)

    for idx, record in enumerate(records):
        source_row = source_rows[record.id]
        source_meta = source_row.get("mobilemem_meta") or {}
        reasoning_type = source_meta.get("reasoning_type")
        if reasoning_type in excluded_reasoning_types:
            audit.append(
                {
                    "id": record.id,
                    "decision": "drop_excluded_reasoning_type",
                    "reasoning_type": reasoning_type,
                }
            )
            checkpoint()
            continue

        clean = pipe.answer(clean_context(record), language=args.language)
        clean_metrics = evaluator.evaluate_one(clean)
        if clean_metrics.contains < 1.0:
            audit.append(
                {
                    "id": record.id,
                    "decision": "drop_clean_wrong",
                    "gold": record.answers_norm,
                    "clean_prediction": clean.prediction,
                    "clean_contains": clean_metrics.contains,
                }
            )
            checkpoint()
            continue

        noise_seed = args.seed + record.id
        noisy = pipe.answer(noisy_context(record, seed=noise_seed), language=args.language)
        noisy_metrics = evaluator.evaluate_one(noisy)
        if noisy_metrics.contains >= 1.0:
            audit.append(
                {
                    "id": record.id,
                    "decision": "drop_noisy_still_correct",
                    "gold": record.answers_norm,
                    "clean_prediction": clean.prediction,
                    "noisy_prediction": noisy.prediction,
                    "noisy_contains": noisy_metrics.contains,
                    "actual_noise_ratio": noisy.noise_ratio,
                }
            )
            checkpoint()
            continue

        row = deepcopy(source_row)
        row["id"] = args.start_id + len(kept)
        meta = row.setdefault("mobilemem_meta", {})
        meta["source_sample_id"] = record.id
        meta["behavior_filter"] = {
            "clean_contains": clean_metrics.contains,
            "clean_prediction": clean.prediction,
            "noisy_contains": noisy_metrics.contains,
            "noisy_prediction": noisy.prediction,
            "actual_noise_ratio": noisy.noise_ratio,
            "target_noise_ratio": 0.95,
            "noise_type": "semantic",
            "noise_position": "interleave",
            "max_docs": 40,
            "min_positive": 2,
            "seed": noise_seed,
            "subset": args.subset,
            "rule": "keep iff clean correct and 95pct-noise wrong",
        }
        kept.append(row)
        audit.append(
            {
                "id": record.id,
                "new_id": row["id"],
                "decision": "keep",
                "gold": record.answers_norm,
                "clean_prediction": clean.prediction,
                "noisy_prediction": noisy.prediction,
                "actual_noise_ratio": noisy.noise_ratio,
            }
        )
        print(
            json.dumps(
                {
                    "seen": args.offset + idx + 1,
                    "kept": len(kept),
                    "id": record.id,
                    "decision": "keep",
                    "clean": clean.prediction,
                    "noisy": noisy.prediction,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        checkpoint()
        if len(kept) >= args.target:
            break

    checkpoint()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": str(audit_path),
                "seen": len(audit),
                "offset": args.offset,
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
