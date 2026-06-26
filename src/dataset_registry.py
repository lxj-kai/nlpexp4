"""Dataset catalog for API / frontend demo."""
from __future__ import annotations

from typing import Any

from .data_loader import Language, Subset

DatasetSpec = dict[str, Any]

DATASETS: dict[str, DatasetSpec] = {
    "rgb": {
        "id": "rgb",
        "label": "RGB（基准 · 新闻/百科）",
        "languages": ["zh", "en"],
        "subsets": ["main", "refine", "fact", "int"],
        "default_language": "zh",
        "default_subset": "main",
        "demo_limit": 300,
    },
    "2wiki": {
        "id": "2wiki",
        "label": "2WikiMultihopQA（英文多跳 · hard neg）",
        "languages": ["en"],
        "subsets": ["main", "fact"],
        "default_language": "en",
        "default_subset": "main",
        "demo_limit": 500,
        "prepare": "python scripts/prepare_2wiki.py",
    },
    "cmedqa": {
        "id": "cmedqa",
        "label": "CmedqaRetrieval（中文医学）",
        "languages": ["zh"],
        "subsets": ["main", "fact"],
        "default_language": "zh",
        "default_subset": "main",
        "demo_limit": 500,
        "prepare": "python scripts/prepare_cmedqa.py",
    },
    "miriad": {
        "id": "miriad",
        "label": "MIRIAD-5.8M（英文医学 · 大规模）",
        "languages": ["en"],
        "subsets": ["main", "fact"],
        "default_language": "en",
        "default_subset": "main",
        "demo_limit": 100,
        "prepare": "python scripts/prepare_miriad.py",
    },
    "bright": {
        "id": "bright",
        "label": "BRIGHT（英文 · hard neg · 长文推理）",
        "languages": ["en"],
        "subsets": ["main", "fact"],
        "default_language": "en",
        "default_subset": "main",
        "demo_limit": 500,
        "prepare": "python scripts/prepare_bright.py",
    },
    "multihop_rag": {
        "id": "multihop_rag",
        "label": "MultiHop-RAG（英文 · evidence 明确 · 新闻整合）",
        "languages": ["en"],
        "subsets": ["main", "fact"],
        "default_language": "en",
        "default_subset": "main",
        "demo_limit": 500,
        "prepare": "python scripts/prepare_multihop_rag.py",
    },
    "tempo": {
        "id": "tempo",
        "label": "TEMPO（英文 · 论坛长文 · 多域）",
        "languages": ["en"],
        "subsets": ["main", "fact"],
        "default_language": "en",
        "default_subset": "main",
        "demo_limit": 500,
        "prepare": "python scripts/prepare_tempo.py",
    },
}


def list_dataset_ids() -> list[str]:
    return list(DATASETS.keys())


def get_dataset_spec(dataset: str) -> DatasetSpec:
    key = dataset.strip().lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return DATASETS[key]


def resolve_demo_params(
    dataset: str,
    language: str | None = None,
    subset: str | None = None,
) -> tuple[Language, Subset]:
    spec = get_dataset_spec(dataset)
    langs: list[str] = spec["languages"]
    subs: list[str] = spec["subsets"]
    lang = (language or spec["default_language"]).strip().lower()
    sub = (subset or spec["default_subset"]).strip().lower()
    if lang not in langs:
        raise ValueError(f"dataset {dataset!r} does not support language {lang!r}")
    if sub not in subs:
        raise ValueError(f"dataset {dataset!r} does not support subset {sub!r}")
    return lang, sub  # type: ignore[return-value]


def demo_limit_for(dataset: str) -> int:
    return int(get_dataset_spec(dataset).get("demo_limit") or 200)


def config_payload() -> dict[str, Any]:
    from src.config import CONFIG

    return {
        "datasets": list(DATASETS.values()),
        "default_dataset": "rgb",
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noise_positions": ["front", "back", "interleave", "surround"],
        "generation_model": CONFIG.model,
        "generation_api_base": CONFIG.api_base,
        "generation_provider": "lmstudio",
        "judge_model": CONFIG.judge_model,
        "judge_api_base": CONFIG.judge_api_base,
        "judge_provider": "deepseek",
    }
