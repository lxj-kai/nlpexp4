"""Shared dependencies — LLM client, evaluator, record cache."""
from __future__ import annotations

from fastapi import HTTPException

from src.data_loader import RGBRecord, load_dataset
from src.noiser_loader import load_noiser_dataset, list_noiser_subsets, normalize_noiser_subset
from src.evaluator import Evaluator
from src.llm_client import LLMClient

llm = LLMClient()
evaluator = Evaluator(use_llm_judge=True, llm=llm)
_records_cache: dict[tuple[str, str], list] = {}

NOISER_SUBSETS = set(list_noiser_subsets()) | {"2wikiimqa"}


def get_records(language: str, subset: str) -> list:
    subset = normalize_noiser_subset(subset)
    key = (language, subset)
    if key not in _records_cache:
        if subset in NOISER_SUBSETS:
            _records_cache[key] = load_noiser_dataset(subset=subset)
        else:
            _records_cache[key] = load_dataset(language=language, subset=subset)
    return _records_cache[key]


def find_record(language: str, subset: str, sample_id: int) -> RGBRecord:
    for r in get_records(language, subset):
        if r.id == sample_id:
            return r
    raise HTTPException(404, detail=f"sample {sample_id} not found")
