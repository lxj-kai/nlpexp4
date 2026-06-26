"""Noise injector — 模拟 RAG 检索上下文：从数据集标注池混合正负文档。

RGB 类数据集每条样本含：
  - positive: 可支撑回答的文档（检索结果中的「好文档」）
  - negative: 语义相关但无法支撑回答的文档（检索噪音，缺逻辑依赖）
  - positive_wrong (fact 子集): 反事实文档（表面像 positive 但内容错误）

NoiserBench 额外提供 7 类预标注噪声池（见 noiser_loader.NOISER_NOISE_TYPES）。

本模块按 noise_ratio / noise_type / noise_position 控制混合方式，
输出 NoisyContext 供 pipeline 拼接后送入 LLM。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from .config import CONFIG
from .data_loader import RGBRecord
from .noiser_loader import NOISER_HARMFUL_TYPES, NOISER_NOISE_TYPES
from .utils import get_logger

logger = get_logger(__name__)

LegacyNoiseType = Literal["semantic", "counterfactual", "mixed"]
NoiserNoiseType = Literal[
    "semantic",
    "counterfactual",
    "supportive",
    "orthographic",
    "datatype",
    "illegal_sentence",
    "counterfactual_answer",
    "mixed",
]
NoiseType = LegacyNoiseType | NoiserNoiseType
NoisePosition = Literal["front", "back", "interleave", "surround"]
DocLabel = Literal[
    "positive",
    "negative",
    "positive_wrong",
    "supportive",
    "orthographic",
    "datatype",
    "illegal_sentence",
    "counterfactual_answer",
]

_NOISER_LABEL_MAP: dict[str, DocLabel] = {
    "semantic": "negative",
    "counterfactual": "positive_wrong",
    "supportive": "supportive",
    "orthographic": "orthographic",
    "datatype": "datatype",
    "illegal_sentence": "illegal_sentence",
    "counterfactual_answer": "counterfactual_answer",
}


@dataclass
class NoisyContext:
    """注入噪音后的上下文样本，供 RAG pipeline 使用。"""

    sample_id: int
    query: str
    gold_answers: list[str]
    docs: list[str]
    labels: list[DocLabel]
    noise_ratio: float
    noise_type: str
    noise_position: NoisePosition
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.docs) == len(self.labels), "docs/labels 长度必须一致"

    @property
    def noise_indices(self) -> list[int]:
        return [i for i, lab in enumerate(self.labels) if lab != "positive"]

    @property
    def positive_indices(self) -> list[int]:
        return [i for i, lab in enumerate(self.labels) if lab == "positive"]


def _uses_noiser_pools(record: RGBRecord) -> bool:
    return record.has_noiser_noises or getattr(record, "dataset", "") == "noiser_bench"


def _noiser_noise_pool(
    record: RGBRecord,
    noise_type: str,
) -> list[tuple[str, DocLabel]]:
    pools = record.noiser_noises or {}
    if noise_type == "mixed":
        out: list[tuple[str, DocLabel]] = []
        for key in NOISER_HARMFUL_TYPES:
            label = _NOISER_LABEL_MAP[key]
            for doc in pools.get(key, []):
                out.append((doc, label))
        return out

    if noise_type not in _NOISER_LABEL_MAP:
        raise ValueError(
            f"未知 NoiserBench noise_type: {noise_type!r}; "
            f"可选: {list(NOISER_NOISE_TYPES)} 或 mixed"
        )

    label = _NOISER_LABEL_MAP[noise_type]
    return [(doc, label) for doc in pools.get(noise_type, []) if doc]


def _noiser_pool_size(record: RGBRecord, noise_type: str) -> int:
    return len(_noiser_noise_pool(record, noise_type))


def _select_noise_docs(
    record: RGBRecord,
    noise_type: str,
    n_noise: int,
    rng: random.Random,
) -> tuple[list[str], list[DocLabel]]:
    """根据 noise_type 从 record 的备选池里选出 n_noise 个噪音文档。"""
    if n_noise <= 0:
        return [], []

    if _uses_noiser_pools(record):
        pool = _noiser_noise_pool(record, noise_type)
    elif noise_type == "semantic":
        pool = [(d, "negative") for d in record.negative]
    elif noise_type == "counterfactual":
        if not record.positive_wrong:
            raise ValueError(
                f"record {record.id} has no positive_wrong docs for counterfactual noise"
            )
        pool = [(d, "positive_wrong") for d in record.positive_wrong]
    elif noise_type == "mixed":
        if not record.positive_wrong:
            raise ValueError(
                f"record {record.id} has no positive_wrong docs for mixed noise"
            )
        pool = [(d, "negative") for d in record.negative] + [
            (d, "positive_wrong") for d in record.positive_wrong
        ]
    else:
        raise ValueError(f"未知 noise_type: {noise_type}")

    if not pool:
        raise ValueError(
            f"record {record.id} has empty noise pool for noise_type={noise_type!r}"
        )

    if n_noise >= len(pool):
        chosen = list(pool)
    else:
        chosen = rng.sample(pool, n_noise)
    docs = [c[0] for c in chosen]
    labels = [c[1] for c in chosen]
    return docs, labels


def _arrange(
    pos_docs: list[str],
    noise_docs: list[str],
    pos_labels: list[DocLabel],
    noise_labels: list[DocLabel],
    position: NoisePosition,
    rng: random.Random,
) -> tuple[list[str], list[DocLabel]]:
    """按 position 策略安排正负文档顺序。"""
    if position == "front":
        return noise_docs + pos_docs, noise_labels + pos_labels
    if position == "back":
        return pos_docs + noise_docs, pos_labels + noise_labels
    if position == "surround":
        if len(noise_docs) < 2:
            return noise_docs + pos_docs, noise_labels + pos_labels
        half = len(noise_docs) // 2
        return (
            noise_docs[:half] + pos_docs + noise_docs[half:],
            noise_labels[:half] + pos_labels + noise_labels[half:],
        )
    # interleave (默认): 随机打散
    docs = pos_docs + noise_docs
    labels = pos_labels + noise_labels
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    return [docs[i] for i in idx], [labels[i] for i in idx]


def inject(
    record: RGBRecord,
    noise_ratio: float,
    *,
    noise_type: str = "semantic",
    noise_position: NoisePosition = "interleave",
    max_docs: int = 10,
    min_positive: int = 1,
    seed: int | None = None,
    dataset: str | None = None,
) -> NoisyContext:
    """对单条记录注入噪音并返回 NoisyContext。"""
    ds = (dataset or getattr(record, "dataset", None) or CONFIG.dataset or "rgb").strip().lower()
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError("noise_ratio 必须 ∈ [0,1]")

    rng = random.Random(seed if seed is not None else record.id * 1000 + int(noise_ratio * 100))

    # 全噪音特殊路径
    if noise_ratio >= 0.999999999:
        noise_docs, noise_labels = _select_noise_docs(record, noise_type, max_docs, rng)
        return NoisyContext(
            sample_id=record.id,
            query=record.query,
            gold_answers=record.answers_norm,
            docs=noise_docs,
            labels=noise_labels,
            noise_ratio=1.0,
            noise_type=noise_type,
            noise_position=noise_position,
            meta={"total": len(noise_docs), "positives": 0, "dataset": ds},
        )

    n_pos_pool = len(record.positive)
    if n_pos_pool == 0:
        raise ValueError(f"record {record.id} 没有 positive 文档，无法构造非全噪音样本")

    if noise_ratio == 0.0:
        n_pos = min(max_docs, n_pos_pool)
        n_noise = 0
    else:
        if _uses_noiser_pools(record):
            pool_size = _noiser_pool_size(record, noise_type)
        else:
            pool_size = len(record.negative) + len(record.positive_wrong)
        approx_total = min(max_docs, n_pos_pool + pool_size)
        n_noise = int(round(approx_total * noise_ratio))
        n_pos = approx_total - n_noise
        n_pos = max(n_pos, min_positive)
        n_pos = min(n_pos, n_pos_pool)
        if n_pos + n_noise > max_docs:
            n_noise = max_docs - n_pos

    pos_sample = rng.sample(record.positive, n_pos)
    pos_labels: list[DocLabel] = ["positive"] * n_pos

    noise_docs, noise_labels = _select_noise_docs(record, noise_type, n_noise, rng)
    docs, labels = _arrange(
        pos_sample, noise_docs, pos_labels, noise_labels, noise_position, rng
    )

    actual_total = len(docs)
    actual_ratio = (actual_total - n_pos) / actual_total if actual_total else 0.0
    return NoisyContext(
        sample_id=record.id,
        query=record.query,
        gold_answers=record.answers_norm,
        docs=docs,
        labels=labels,
        noise_ratio=round(actual_ratio, 3),
        noise_type=noise_type,
        noise_position=noise_position,
        meta={
            "total": actual_total,
            "positives": n_pos,
            "noises": len(noise_docs),
            "dataset": ds,
            "subset": getattr(record, "subset", None),
        },
    )


def batch_inject(records: list[RGBRecord], **kwargs) -> list[NoisyContext]:
    """批量注入；任何无法构造的样本直接抛错。"""
    return [inject(r, **kwargs) for r in records]


def batch_closed_book(records: list[RGBRecord], *, dataset: str | None = None) -> list[NoisyContext]:
    """构造 closed-book 上下文：只保留问题与 gold，不提供任何文档。"""
    default_ds = (dataset or CONFIG.dataset or "rgb").strip().lower()
    return [
        NoisyContext(
            sample_id=record.id,
            query=record.query,
            gold_answers=record.answers_norm,
            docs=[],
            labels=[],
            noise_ratio=0.0,
            noise_type="semantic",
            noise_position="interleave",
            meta={
                "closed_book": True,
                "total": 0,
                "positives": 0,
                "dataset": (dataset or getattr(record, "dataset", None) or default_ds),
            },
        )
        for record in records
    ]


def noise_types_for_dataset(dataset: str) -> list[str]:
    """返回某数据集可用的 noise_type 列表。"""
    ds = dataset.strip().lower()
    if ds == "noiser_bench":
        return [*NOISER_NOISE_TYPES, "mixed"]
    return ["semantic", "counterfactual", "mixed"]
