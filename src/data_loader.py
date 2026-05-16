"""RGB dataset loader.

数据文件格式（JSON Lines, 每行一条记录）：

- zh.json / zh_refine.json / en.json / en_refine.json
    {id, query, answer:list, positive:list[str], negative:list[str]}
- zh_fact.json / en_fact.json (反事实)
    {id, query, answer:str, fakeanswer:str, positive_wrong:list[str],
     positive:list[str], negative:list[str]}
- zh_int.json / en_int.json (信息整合)
    {id, query, answer:list, asnwer1/answer2:list, positive, negative}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from .config import CONFIG
from .utils import get_logger

logger = get_logger(__name__)

Language = Literal["zh", "en"]
Subset = Literal["main", "refine", "fact", "int"]


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

    @property
    def has_counterfactual(self) -> bool:
        return bool(self.positive_wrong)

    @property
    def answers_norm(self) -> list[str]:
        """规范化的答案列表（过滤空值）。"""
        return [a for a in self.answer if a]


def _coerce_answer(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if raw is None:
        return []
    return [str(raw)]


def parse_record(raw: dict, *, language: Language, subset: Subset) -> RGBRecord:
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


def iter_records(
    language: Language = "zh", subset: Subset = "main"
) -> Iterator[RGBRecord]:
    fname = _FILE_MAP[(language, subset)]
    path = CONFIG.data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"RGB data not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"skip malformed line {idx} in {fname}: {e}")
                continue
            yield parse_record(raw, language=language, subset=subset)


def load_dataset(
    language: Language = "zh",
    subset: Subset = "main",
    *,
    limit: int | None = None,
) -> list[RGBRecord]:
    records: list[RGBRecord] = []
    for rec in iter_records(language=language, subset=subset):
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    logger.info(f"loaded {len(records)} records from {language}/{subset}")
    return records


def load_all_subsets(language: Language = "zh") -> dict[Subset, list[RGBRecord]]:
    return {
        "main": load_dataset(language, "main"),
        "refine": load_dataset(language, "refine"),
        "fact": load_dataset(language, "fact"),
        "int": load_dataset(language, "int"),
    }
