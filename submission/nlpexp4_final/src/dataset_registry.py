"""Dataset catalog for API / frontend demo."""
from __future__ import annotations

from typing import Any

from .data_loader import Language, Subset
from .noiser_loader import NOISER_NOISE_TYPES

DatasetSpec = dict[str, Any]

# strategyqa 为 yes/no + 强参数知识，不适合本项目「检索噪音鲁棒性」实验（加噪仍易答对）
NOISER_BENCH_EXCLUDED_SUBSETS: frozenset[str] = frozenset({"strategyqa"})

NOISER_BENCH_SUBSETS = (
    "nq",
    "rgb_nb",
    "hotpotqa",
    "2wikimqa",
    "bamboogle",
    "tempqa",
    "priorqa",
)

NOISER_BENCH_ALL_SUBSETS = (*NOISER_BENCH_SUBSETS, *NOISER_BENCH_EXCLUDED_SUBSETS)

NOISER_BENCH_NOISE_TYPES = [*NOISER_NOISE_TYPES, "mixed"]


def noiser_bench_subsets_available(*, include_excluded: bool = False) -> list[str]:
    from .noiser_loader import list_noiser_subsets

    found = list_noiser_subsets()
    base = found if found else list(NOISER_BENCH_ALL_SUBSETS if include_excluded else NOISER_BENCH_SUBSETS)
    if include_excluded:
        return base
    return [s for s in base if s not in NOISER_BENCH_EXCLUDED_SUBSETS]


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
    "noiser_bench": {
        "id": "noiser_bench",
        "label": "NoiserBench（ACL'25 · 7类噪音 RAG）",
        "languages": ["en"],
        "subsets": list(NOISER_BENCH_SUBSETS),
        "default_language": "en",
        "default_subset": "hotpotqa",
        "demo_limit": 300,
        "prepare": "python scripts/prepare_noiser_bench.py",
        "noise_types": NOISER_BENCH_NOISE_TYPES,
        "notes": "推荐 hotpotqa/2wikimqa/nq；strategyqa(yes/no) 已排除，不适合测检索噪音",
    },
    "mobilemem": {
        "id": "mobilemem",
        "label": "MobileMem（中文 · 购物截图 OCR 记忆图谱）",
        "languages": ["zh"],
        "subsets": ["calc", "noncalc"],
        "default_language": "zh",
        "default_subset": "calc",
        "demo_limit": 120,
        "prepare": "python scripts/prepare_mobilemem.py",
        "notes": "购物截图 OCR 文本证据；calc=多事件聚合计算，noncalc=检索问答",
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
    ds = dataset.strip().lower()
    if ds == "mobilemem":
        sub = {"mobilemem_calc": "calc", "mobilemem_noncalc": "noncalc"}.get(sub, sub)
    if lang not in langs:
        raise ValueError(f"dataset {dataset!r} does not support language {lang!r}")
    if sub not in subs:
        raise ValueError(f"dataset {dataset!r} does not support subset {sub!r}")
    if ds == "noiser_bench" and sub in NOISER_BENCH_EXCLUDED_SUBSETS:
        raise ValueError(
            f"subset {sub!r} is excluded from noise-robustness demo "
            f"(yes/no StrategyQA: models often answer from parametric knowledge). "
            f"Use: {list(NOISER_BENCH_SUBSETS)}"
        )
    return lang, sub  # type: ignore[return-value]


def demo_limit_for(dataset: str) -> int:
    return int(get_dataset_spec(dataset).get("demo_limit") or 200)


def config_payload() -> dict[str, Any]:
    from src.config import CONFIG

    datasets = []
    for spec in DATASETS.values():
        entry = dict(spec)
        entry.setdefault("noise_types", ["semantic", "counterfactual", "mixed"])
        if entry["id"] == "noiser_bench":
            entry["subsets"] = noiser_bench_subsets_available()
        datasets.append(entry)

    return {
        "datasets": datasets,
        "default_dataset": "rgb",
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noiser_noise_types": NOISER_BENCH_NOISE_TYPES,
        "noise_positions": ["front", "back", "interleave", "surround"],
        "generation_model": CONFIG.model,
        "generation_api_base": CONFIG.api_base,
        "generation_provider": "lmstudio",
        "judge_model": CONFIG.judge_model,
        "judge_api_base": CONFIG.judge_api_base,
        "judge_provider": "deepseek",
    }
