"""RGB dataset loader.

数据文件格式（JSON Lines, 每行一条记录）：

- zh.json / zh_refine.json / en.json / en_refine.json
    {id, query, answer:list, positive:list[str], negative:list[str]}
- zh_fact.json / en_fact.json (反事实)
    {id, query, answer:str, fakeanswer:str, positive_wrong:list[str],
     positive:list[str], negative:list[str]}
- zh_int.json / en_int.json (信息整合)
    {id, query, answer:list, asnwer1/answer2:list, positive, negative}

MIRIAD 全量数据位于 data/miriad/raw/（64 个 parquet，约 7.5GB），由
scripts/prepare_miriad.py 完整下载；运行时由 src/miriad_store.py 直接读取。

CmedqaRetrieval 子集（data/cmedqa/）提供 zh.json / zh_fact.json，由
scripts/prepare_cmedqa.py 从 HuggingFace 转换而来。

2WikiMultihopQA（data/2wiki/）提供 en.json / en_fact.json，由
scripts/prepare_2wiki.py 从 xanhho/2WikiMultihopQA 转换而来；负例为
同 context 内的 distractor 段落（hard negative）。

BRIGHT（data/bright/）、MultiHop-RAG（data/multihop_rag/）、TEMPO（data/tempo/）
提供 en.json / en_fact.json，分别由 scripts/prepare_bright.py、
scripts/prepare_multihop_rag.py、scripts/prepare_tempo.py 从 HuggingFace 转换。

MobileMem（data/mobilemem/）提供 zh_calc.json / zh_noncalc.json，购物截图 OCR
记忆图谱 RAG 子集，由 scripts/prepare_mobilemem.py 从仓库数据整理而来。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from .config import CONFIG, resolve_data_dir
from .utils import get_logger

logger = get_logger(__name__)

Language = Literal["zh", "en"]
Subset = Literal[
    "main",
    "refine",
    "fact",
    "int",
    "calc",
    "noncalc",
]
DatasetName = Literal[
    "rgb",
    "miriad",
    "cmedqa",
    "2wiki",
    "bright",
    "multihop_rag",
    "tempo",
    "noiser_bench",
    "mobilemem",
]


@dataclass
class RGBRecord:
    """RGB 原始记录的标准化容器。"""

    id: int
    query: str
    answer: list[str]
    positive: list[str]
    negative: list[str]
    positive_wrong: list[str] = field(default_factory=list)
    fakeanswer: str = ""
    language: Language = "zh"
    subset: Subset = "main"
    dataset: DatasetName = "rgb"
    # NoiserBench: 7 类预标注噪声池（key → 文档列表）
    noiser_noises: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_counterfactual(self) -> bool:
        return bool(self.positive_wrong)

    @property
    def has_noiser_noises(self) -> bool:
        return bool(self.noiser_noises)

    @property
    def answers_norm(self) -> list[str]:
        """规范化的答案列表（过滤空值；保留 False/no 等合法答案）。"""
        out: list[str] = []
        for a in self.answer:
            if a is None:
                continue
            if isinstance(a, bool):
                out.append("yes" if a else "no")
                continue
            s = str(a).strip()
            if s:
                out.append(s)
        return out


def _answer_to_str(item: object) -> str:
    if isinstance(item, bool):
        return "yes" if item else "no"
    return str(item)


def is_usable_gold_answer(answer: object) -> bool:
    """Reject empty / N/A placeholders (common in BRIGHT code/math subdomains)."""
    if answer is None:
        return False
    if isinstance(answer, bool):
        return True
    s = str(answer).strip()
    if not s:
        return False
    return s.upper() not in ("N/A", "NA")


def record_has_usable_gold(record: RGBRecord) -> bool:
    return bool(record.answers_norm) and all(
        is_usable_gold_answer(a) for a in record.answers_norm
    )


def _coerce_answer(raw: object) -> list[str]:
    if isinstance(raw, bool):
        return [_answer_to_str(raw)]
    if isinstance(raw, list):
        return [_answer_to_str(a) for a in raw]
    if raw is None:
        return []
    return [_answer_to_str(raw)]


def parse_record(raw: dict, *, language: Language, subset: Subset, dataset: DatasetName = "rgb") -> RGBRecord:
    return RGBRecord(
        id=int(raw["id"]),
        query=str(raw["query"]),
        answer=_coerce_answer(raw.get("answer")),
        positive=[str(x) for x in raw.get("positive", [])],
        negative=[str(x) for x in raw.get("negative", [])],
        positive_wrong=[str(x) for x in raw.get("positive_wrong", [])],
        fakeanswer=str(raw.get("fakeanswer", "")),
        language=language,
        subset=subset,
        dataset=dataset,
    )


_FILE_MAP: dict[tuple[Language, Subset], str] = {
    ("zh", "main"): "zh.json",
    ("zh", "refine"): "zh_refine.json",
    ("zh", "fact"): "zh_fact.json",
    ("zh", "int"): "zh_int.json",
    ("en", "main"): "en.json",
    ("en", "refine"): "en_refine.json",
    ("en", "fact"): "en_fact.json",
    ("en", "int"): "en_int.json",
}

_MOBILEMEM_FILE_MAP: dict[tuple[Language, Subset], str] = {
    ("zh", "calc"): "zh_calc.json",
    ("zh", "noncalc"): "zh_noncalc.json",
}

_MOBILEMEM_SUPPORTED: set[tuple[Language, Subset]] = {
    ("zh", "calc"),
    ("zh", "noncalc"),
}

_MIRIAD_SUPPORTED: set[tuple[Language, Subset]] = {
    ("en", "main"),
    ("en", "fact"),
}


_CMEDQA_SUPPORTED: set[tuple[Language, Subset]] = {
    ("zh", "main"),
    ("zh", "fact"),
}

_2WIKI_SUPPORTED: set[tuple[Language, Subset]] = {
    ("en", "main"),
    ("en", "fact"),
}

_EN_FACT_SUPPORTED: set[tuple[Language, Subset]] = {
    ("en", "main"),
    ("en", "fact"),
}


def _resolve_dataset(dataset: DatasetName | None) -> DatasetName:
    name = (dataset or CONFIG.dataset or "rgb").strip().lower()
    if name not in (
        "rgb",
        "miriad",
        "cmedqa",
        "2wiki",
        "bright",
        "multihop_rag",
        "tempo",
        "noiser_bench",
        "mobilemem",
    ):
        raise ValueError(f"unknown dataset: {name!r}")
    return name  # type: ignore[return-value]


def _data_dir_for(dataset: DatasetName) -> Path:
    return resolve_data_dir(dataset)


def iter_records(
    language: Language = "zh",
    subset: Subset = "main",
    *,
    dataset: DatasetName | None = None,
    limit: int | None = None,
) -> Iterator[RGBRecord]:
    ds = _resolve_dataset(dataset)
    if ds == "miriad":
        if (language, subset) not in _MIRIAD_SUPPORTED:
            raise ValueError(
                f"MIRIAD only supports en/main and en/fact, got {language}/{subset}"
            )
        if limit is None:
            raise ValueError("MIRIAD requires explicit limit=... (full 5.8M load is forbidden)")
        from .miriad_store import iter_miriad_records

        subset_name = "main" if subset == "main" else "fact"
        yield from iter_miriad_records(
            subset=subset_name,  # type: ignore[arg-type]
            shuffle=False,
            seed=CONFIG.seed,
            limit=limit,
        )
        return
    if ds == "noiser_bench":
        if language != "en":
            raise ValueError(f"NoiserBench only supports en, got {language!r}")
        from .noiser_loader import iter_noiser_records

        nb_subset = str(subset)
        for rec in iter_noiser_records(nb_subset):
            yield rec
        return
    if ds == "cmedqa":
        if (language, subset) not in _CMEDQA_SUPPORTED:
            raise ValueError(
                f"CmedqaRetrieval only supports zh/main and zh/fact, got {language}/{subset}"
            )
    if ds == "2wiki":
        if (language, subset) not in _2WIKI_SUPPORTED:
            raise ValueError(
                f"2WikiMultihopQA only supports en/main and en/fact, got {language}/{subset}"
            )
    if ds in ("bright", "multihop_rag", "tempo"):
        if (language, subset) not in _EN_FACT_SUPPORTED:
            raise ValueError(
                f"{ds} only supports en/main and en/fact, got {language}/{subset}"
            )
    if ds == "mobilemem":
        if (language, subset) not in _MOBILEMEM_SUPPORTED:
            raise ValueError(
                f"MobileMem only supports zh/calc and zh/noncalc, got {language}/{subset}"
            )
        fname = _MOBILEMEM_FILE_MAP[(language, subset)]
        path = _data_dir_for(ds) / fname
    else:
        fname = _FILE_MAP[(language, subset)]
        path = _data_dir_for(ds) / fname
    if not path.exists():
        _PREPARE_HINTS = {
            "cmedqa": "python scripts/prepare_cmedqa.py",
            "2wiki": "python scripts/prepare_2wiki.py",
            "bright": "python scripts/prepare_bright.py",
            "multihop_rag": "python scripts/prepare_multihop_rag.py",
            "tempo": "python scripts/prepare_tempo.py",
            "noiser_bench": "python scripts/prepare_noiser_bench.py",
            "mobilemem": "python scripts/prepare_mobilemem.py",
        }
        hint = _PREPARE_HINTS.get(ds)
        if hint:
            raise FileNotFoundError(
                f"dataset file not found: {path}. Run `{hint}` first."
            )
        raise FileNotFoundError(f"dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield parse_record(raw, language=language, subset=subset, dataset=ds)


def load_dataset(
    language: Language = "zh",
    subset: Subset = "main",
    *,
    dataset: DatasetName | None = None,
    limit: int | None = None,
    shuffle: bool = True,
) -> list[RGBRecord]:
    ds = _resolve_dataset(dataset)
    if ds == "miriad":
        if (language, subset) not in _MIRIAD_SUPPORTED:
            raise ValueError(
                f"MIRIAD only supports en/main and en/fact, got {language}/{subset}"
            )
        if limit is None:
            raise ValueError("MIRIAD requires explicit limit=... (full 5.8M load is forbidden)")
        from .miriad_store import load_miriad_records

        subset_name = "main" if subset == "main" else "fact"
        records = load_miriad_records(
            subset=subset_name,  # type: ignore[arg-type]
            limit=limit,
            shuffle=shuffle,
            seed=CONFIG.seed,
        )
        logger.info(
            f"loaded {len(records)} records from {ds}/{language}/{subset} (shuffle={shuffle})"
        )
        return records

    records: list[RGBRecord] = list(
        iter_records(language=language, subset=subset, dataset=ds, limit=limit)
    )
    if shuffle:
        import random as _rng
        _rng.Random(CONFIG.seed).shuffle(records)
    if limit is not None:
        records = records[:limit]
    logger.info(
        f"loaded {len(records)} records from {ds}/{language}/{subset} (shuffle={shuffle})"
    )
    return records


def load_all_subsets(
    language: Language = "zh", *, dataset: DatasetName | None = None
) -> dict[Subset, list[RGBRecord]]:
    ds = _resolve_dataset(dataset)
    if ds in ("miriad", "cmedqa", "2wiki", "bright", "multihop_rag", "tempo", "noiser_bench"):
        raise ValueError(
            f"{ds} only supports main/fact via explicit load_dataset(..., subset=...); "
            "load_all_subsets is not supported"
        )
    return {
        "main": load_dataset(language, "main", dataset=ds),
        "refine": load_dataset(language, "refine", dataset=ds),
        "fact": load_dataset(language, "fact", dataset=ds),
        "int": load_dataset(language, "int", dataset=ds),
    }
