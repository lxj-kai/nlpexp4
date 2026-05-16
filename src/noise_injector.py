"""Noise injector — 控制实验的核心变量：噪音比例 / 类型 / 位置。"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from .data_loader import RGBRecord
from .utils import get_logger

logger = get_logger(__name__)

NoiseType = Literal["semantic", "counterfactual", "mixed"]
NoisePosition = Literal["front", "back", "interleave", "surround"]
DocLabel = Literal["positive", "negative", "positive_wrong"]


@dataclass
class NoisyContext:
    """注入噪音后的上下文样本，供 RAG pipeline 使用。"""

    sample_id: int
    query: str
    gold_answers: list[str]
    docs: list[str]
    labels: list[DocLabel]
    noise_ratio: float
    noise_type: NoiseType
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


def _select_noise_docs(
    record: RGBRecord,
    noise_type: NoiseType,
    n_noise: int,
    rng: random.Random,
) -> tuple[list[str], list[DocLabel]]:
    """根据 noise_type 从 record 的备选池里选出 n_noise 个噪音文档。"""
    if n_noise <= 0:
        return [], []

    pool: list[tuple[str, DocLabel]] = []
    if noise_type == "semantic":
        pool = [(d, "negative") for d in record.negative]
    elif noise_type == "counterfactual":
        if record.positive_wrong:
            pool = [(d, "positive_wrong") for d in record.positive_wrong]
        else:
            logger.debug(
                f"record {record.id} 无 positive_wrong, 回退到 semantic 噪音"
            )
            pool = [(d, "negative") for d in record.negative]
    elif noise_type == "mixed":
        pool = [(d, "negative") for d in record.negative] + [
            (d, "positive_wrong") for d in record.positive_wrong
        ]
    else:
        raise ValueError(f"未知 noise_type: {noise_type}")

    if not pool:
        return [], []

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
    noise_type: NoiseType = "semantic",
    noise_position: NoisePosition = "interleave",
    max_docs: int = 10,
    min_positive: int = 1,
    seed: int | None = None,
) -> NoisyContext:
    """对单条记录注入噪音并返回 NoisyContext。"""
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
            meta={"total": len(noise_docs), "positives": 0},
        )

    n_pos_pool = len(record.positive)
    if n_pos_pool == 0:
        raise ValueError(f"record {record.id} 没有 positive 文档，无法构造非全噪音样本")

    if noise_ratio == 0.0:
        n_pos = min(max_docs, n_pos_pool)
        n_noise = 0
    else:
        approx_total = min(max_docs, n_pos_pool + len(record.negative) + len(record.positive_wrong))
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
        },
    )


def batch_inject(records: list[RGBRecord], **kwargs) -> list[NoisyContext]:
    """批量注入；遇到无法构造的样本会跳过并 warn。"""
    out: list[NoisyContext] = []
    for r in records:
        try:
            out.append(inject(r, **kwargs))
        except ValueError as e:
            logger.warning(f"skip record {r.id}: {e}")
    return out
