"""对比 baseline — Self-RAG 思路（简化复现）。

核心：用 LLM 自反思的"相关 / 支撑"判断作为门控信号：
1. RELEVANT 判定：剔除被判为 IRRELEVANT 的文档；
2. 生成答案；
3. SUPPORT 判定：如果答案被判 UNSUPPORTED，则重新仅基于 RELEVANT-top 文档再生成一次。

API 调用：n_docs + 1 (gen) + 1 (support) + 可选 1 (re-gen) = 通常 n+2 ~ n+3。
"""
from __future__ import annotations

from ..noise_injector import NoisyContext
from ..prompts import (
    NAIVE_SYSTEM_ZH,
    NAIVE_USER_TMPL,
    SELFRAG_REL_SYSTEM_ZH,
    SELFRAG_REL_USER_TMPL,
    SELFRAG_SUPPORT_SYSTEM_ZH,
    SELFRAG_SUPPORT_USER_TMPL,
    format_context,
)
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


def _norm_flag(text: str, candidates: tuple[str, ...]) -> str:
    t = (text or "").strip().upper()
    for c in candidates:
        if c in t:
            return c
    return candidates[0]


@register_corrector("selfrag")
class SelfRAGBaseline(BaseCorrector):
    """Self-RAG 思路的对比方法实现。"""

    api_cost = 0

    def _judge_relevance(self, query: str, doc: str) -> bool:
        out = self.llm.chat(
            [
                {"role": "system", "content": SELFRAG_REL_SYSTEM_ZH},
                {
                    "role": "user",
                    "content": SELFRAG_REL_USER_TMPL.format(query=query, doc=doc[:1500]),
                },
            ],
            max_tokens=4,
        )
        flag = _norm_flag(out["content"], ("RELEVANT", "IRRELEVANT"))
        return flag == "RELEVANT"

    def _generate(self, query: str, docs: list[str]) -> dict:
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
        return self.llm.chat(
            [
                {"role": "system", "content": NAIVE_SYSTEM_ZH},
                {"role": "user", "content": user},
            ]
        )

    def _judge_support(self, query: str, answer: str, docs: list[str]) -> str:
        out = self.llm.chat(
            [
                {"role": "system", "content": SELFRAG_SUPPORT_SYSTEM_ZH},
                {
                    "role": "user",
                    "content": SELFRAG_SUPPORT_USER_TMPL.format(
                        query=query, answer=answer, context=format_context(docs)
                    ),
                },
            ],
            max_tokens=4,
        )
        return _norm_flag(out["content"], ("SUPPORTED", "PARTIAL", "UNSUPPORTED"))

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        flags = [self._judge_relevance(ctx.query, d) for d in ctx.docs]
        relevant = [(i, d, l) for i, (d, l, f) in enumerate(zip(ctx.docs, ctx.labels, flags)) if f]
        if not relevant:
            relevant = list(zip(range(len(ctx.docs)), ctx.docs, ctx.labels))

        rel_idx = [i for i, _, _ in relevant]
        rel_docs = [d for _, d, _ in relevant]
        rel_labels = [l for _, _, l in relevant]

        gen = self._generate(ctx.query, rel_docs)
        answer = (gen["content"] or "").strip()

        support = self._judge_support(ctx.query, answer, rel_docs)
        api_calls = len(ctx.docs) + 2

        if support == "UNSUPPORTED" and len(rel_docs) > 1:
            top_docs = rel_docs[: max(1, len(rel_docs) // 2)]
            gen2 = self._generate(ctx.query, top_docs)
            answer = (gen2["content"] or "").strip()
            api_calls += 1

        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=answer,
            docs=rel_docs,
            labels=list(rel_labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "relevant_flags": flags,
                "kept_indices": rel_idx,
                "support_flag": support,
                "api_calls": api_calls,
                "prompt_tokens": gen.get("prompt_tokens", 0),
                "completion_tokens": gen.get("completion_tokens", 0),
                **ctx.meta,
            },
        )
