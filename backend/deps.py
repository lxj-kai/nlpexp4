"""Shared dependencies — LLM client, evaluator, record cache."""
from __future__ import annotations

from fastapi import HTTPException

from src.config import CONFIG
from src.data_loader import RGBRecord, load_dataset
from src.dataset_registry import demo_limit_for, get_dataset_spec, resolve_demo_params
from src.evaluator import Evaluator
from src.llm_client import LLMClient, get_judge_client

llm = LLMClient()
judge_llm = get_judge_client()
evaluator = Evaluator(
    use_llm_judge=True,
    use_legacy_metrics=False,
    use_semantic_attribution=False,
    llm=llm,
    judge_llm=judge_llm,
)
_records_cache: dict[tuple[str, str, str], list[RGBRecord]] = {}


def get_records(dataset: str, language: str, subset: str) -> list[RGBRecord]:
    ds = dataset.strip().lower()
    try:
        lang, sub = resolve_demo_params(ds, language, subset)
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e

    key = (ds, lang, sub)
    if key not in _records_cache:
        limit = demo_limit_for(ds)
        try:
            _records_cache[key] = load_dataset(
                language=lang,
                subset=sub,
                dataset=ds,  # type: ignore[arg-type]
                limit=limit,
                shuffle=True,
            )
        except FileNotFoundError as e:
            spec = get_dataset_spec(ds)
            hint = spec.get("prepare", "请检查 data/ 目录下是否有对应 JSONL")
            raise HTTPException(
                404,
                detail=f"数据集 {ds!r} 尚未准备：{e}. 运行: {hint}",
            ) from e
    return _records_cache[key]


def find_record(dataset: str, language: str, subset: str, sample_id: int) -> RGBRecord:
    for r in get_records(dataset, language, subset):
        if r.id == sample_id:
            return r
    raise HTTPException(404, detail=f"sample {sample_id} not found in {dataset}/{language}/{subset}")
