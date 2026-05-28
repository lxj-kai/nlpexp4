"""NoiserBench dataset loader.

Loads data from the NoiserBench benchmark (ACL 2025).
Converts to RGBRecord format for compatibility with existing pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .config import CONFIG
from .data_loader import RGBRecord
from .utils import get_logger

logger = get_logger(__name__)

NOISER_DIR = CONFIG.project_root / "data" / "noiser_bench"

_SUBSET_MAP = {
    "nq": "single-hop/nq.json",
    "rgb_nb": "single-hop/rgb.json",
    "hotpotqa": "multi-hop/explicit/hotpotqa.json",
    "2wikimqa": "multi-hop/explicit/2wikimqa.json",
    "bamboogle": "multi-hop/explicit/bamboogle.json",
    "strategyqa": "multi-hop/implicit/strategyqa.json",
    "tempqa": "multi-hop/implicit/tempqa.json",
    "priorqa": "mix-hop/priorqa.json",
}


def _extract_noise_docs(raw: dict, noise_key: str) -> list[str]:
    val = raw.get(noise_key, [])
    if isinstance(val, list):
        return [item["text"] if isinstance(item, dict) else str(item) for item in val]
    if isinstance(val, str) and val:
        return [val]
    if isinstance(val, dict):
        return [str(v) for v in val.values() if v]
    return []


def parse_noiser_record(raw: dict, idx: int, subset: str) -> RGBRecord:
    gold_text = raw.get("gold_text", "")
    positive = [gold_text] if gold_text else []

    answers_raw = raw.get("gold_answers", [])
    if isinstance(answers_raw, str):
        answers_raw = [answers_raw]

    semantic = _extract_noise_docs(raw, "semantic noise")
    counterfactual_text = raw.get("counterfactual noise", "")
    counterfactual_text2 = raw.get("counterfactual noise2", "")
    positive_wrong = [t for t in [counterfactual_text, counterfactual_text2] if isinstance(t, str) and t]

    supportive = _extract_noise_docs(raw, "supportive noise")
    negative = semantic + supportive

    return RGBRecord(
        id=raw.get("id", idx),
        query=raw.get("question", ""),
        answer=answers_raw,
        positive=positive,
        negative=negative,
        positive_wrong=positive_wrong,
        fakeanswer=raw.get("counterfactual answer", ""),
        language="en",
        subset=subset,
    )


def iter_noiser_records(subset: str = "nq") -> Iterator[RGBRecord]:
    fname = _SUBSET_MAP.get(subset)
    if not fname:
        raise ValueError(f"Unknown NoiserBench subset: {subset}. Available: {list(_SUBSET_MAP)}")

    path = NOISER_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"NoiserBench data not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, raw in enumerate(data):
        yield parse_noiser_record(raw, idx, subset)


def load_noiser_dataset(
    subset: str = "nq",
    *,
    limit: int | None = None,
    shuffle: bool = True,
) -> list[RGBRecord]:
    records = list(iter_noiser_records(subset))
    if shuffle:
        import random as _rng
        _rng.Random(CONFIG.seed).shuffle(records)
    if limit is not None:
        records = records[:limit]
    logger.info(f"loaded {len(records)} NoiserBench records from {subset}")
    return records


def list_noiser_subsets() -> list[str]:
    available = []
    for key, fname in _SUBSET_MAP.items():
        if (NOISER_DIR / fname).exists():
            available.append(key)
    return sorted(available)
