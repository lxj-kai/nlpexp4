"""方法 B — Iterative filter-then-generate.

Step1: 用 LLM 评估每篇文档相关性 (high/mid/low)
Step2: 过滤低相关文档
Step3: 基于剩余文档生成答案

API 调用：n_docs + 1 次（每文档 1 次过滤 + 1 次最终生成）。
"""
from __future__ import annotations

from ..config import CONFIG
from ..noise_injector import NoisyContext
from ..prompts import (
    ITER_FILTER_SYSTEM_EN,
    ITER_FILTER_SYSTEM_ZH,
    ITER_FILTER_USER_TMPL,
    ITER_FILTER_USER_TMPL_EN,
    NAIVE_SYSTEM_EN,
    NAIVE_SYSTEM_ZH,
    NAIVE_USER_TMPL,
    format_context,
)
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


def _score_label(label: str) -> int:
    """high=2 / mid=1 / low=0，便于排序与阈值。"""
    label = label.lower()
    if "high" in label:
        return 2
    if "mid" in label:
        return 1
    return 0


@register_corrector("iterative")
class IterativeCorrector(BaseCorrector):
    """方法 B：过滤式两阶段矫正。"""

    api_cost = 0  # 实际由 batch 时动态统计

    def __init__(self, *, keep_min: int = 1, keep_threshold: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keep_min = keep_min
        self.keep_threshold = keep_threshold

    def _rate_doc(self, query: str, doc: str, *, language: str = "zh") -> int:
        sys_msg = ITER_FILTER_SYSTEM_ZH if language == "zh" else ITER_FILTER_SYSTEM_EN
        tmpl = ITER_FILTER_USER_TMPL if language == "zh" else ITER_FILTER_USER_TMPL_EN
        out = self.llm.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": tmpl.format(query=query, doc=doc[:1500])},
            ],
            max_tokens=8,
        )
        return _score_label((out["content"] or "").strip())

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        scores = [self._rate_doc(ctx.query, d, language=language) for d in ctx.docs]

        kept = [(i, s) for i, s in enumerate(scores) if s >= self.keep_threshold]
        if len(kept) < self.keep_min:
            kept = sorted(enumerate(scores), key=lambda x: -x[1])[: self.keep_min]
        kept.sort(key=lambda x: x[0])
        keep_idx = [i for i, _ in kept]

        filt_docs = [ctx.docs[i] for i in keep_idx]
        filt_labels = [ctx.labels[i] for i in keep_idx]

        sys_gen = NAIVE_SYSTEM_ZH if language == "zh" else NAIVE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(
            query=ctx.query, n=len(filt_docs), context=format_context(filt_docs)
        )
        out = self.llm.chat(
            [
                {"role": "system", "content": sys_gen},
                {"role": "user", "content": user},
            ]
        )
        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=out["content"].strip(),
            docs=filt_docs,
            labels=list(filt_labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "doc_scores": scores,
                "kept_indices": keep_idx,
                "kept_n": len(keep_idx),
                "api_calls": len(ctx.docs) + 1,
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "latency": out.get("latency", 0.0),
                **ctx.meta,
            },
        )
