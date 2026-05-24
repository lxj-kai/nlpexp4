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
    NAIVE_SYSTEM_EN,
    NAIVE_SYSTEM_ZH,
    NAIVE_USER_TMPL,
    SELFRAG_REL_SYSTEM_EN,
    SELFRAG_REL_SYSTEM_ZH,
    SELFRAG_REL_USER_TMPL,
    SELFRAG_REL_USER_TMPL_EN,
    SELFRAG_SUPPORT_SYSTEM_EN,
    SELFRAG_SUPPORT_SYSTEM_ZH,
    SELFRAG_SUPPORT_USER_TMPL,
    SELFRAG_SUPPORT_USER_TMPL_EN,
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

    def _judge_relevance(self, query: str, doc: str, *, language: str = "zh") -> bool:
        sys_msg = SELFRAG_REL_SYSTEM_ZH if language == "zh" else SELFRAG_REL_SYSTEM_EN
        tmpl = SELFRAG_REL_USER_TMPL if language == "zh" else SELFRAG_REL_USER_TMPL_EN
        out = self.llm.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": tmpl.format(query=query, doc=doc[:1500])},
            ],
            max_tokens=4,
        )
        flag = _norm_flag(out["content"], ("RELEVANT", "IRRELEVANT"))
        return flag == "RELEVANT"

    def _generate(self, query: str, docs: list[str], *, language: str = "zh") -> dict:
        sys_msg = NAIVE_SYSTEM_ZH if language == "zh" else NAIVE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
        return self.llm.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
            ]
        )

    def _judge_support(self, query: str, answer: str, docs: list[str], *, language: str = "zh") -> str:
        sys_msg = SELFRAG_SUPPORT_SYSTEM_ZH if language == "zh" else SELFRAG_SUPPORT_SYSTEM_EN
        tmpl = SELFRAG_SUPPORT_USER_TMPL if language == "zh" else SELFRAG_SUPPORT_USER_TMPL_EN
        out = self.llm.chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": tmpl.format(query=query, answer=answer, context=format_context(docs))},
            ],
            max_tokens=4,
        )
        return _norm_flag(out["content"], ("SUPPORTED", "PARTIAL", "UNSUPPORTED"))

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        flags = [self._judge_relevance(ctx.query, d, language=language) for d in ctx.docs]
        relevant = [(i, d, l) for i, (d, l, f) in enumerate(zip(ctx.docs, ctx.labels, flags)) if f]
        fallback_to_all = not relevant
        if fallback_to_all:
            relevant = list(zip(range(len(ctx.docs)), ctx.docs, ctx.labels))

        rel_idx = [i for i, _, _ in relevant]
        rel_docs = [d for _, d, _ in relevant]
        rel_labels = [l for _, _, l in relevant]

        gen = self._generate(ctx.query, rel_docs, language=language)
        answer = (gen["content"] or "").strip()

        support = self._judge_support(ctx.query, answer, rel_docs, language=language)
        api_calls = len(ctx.docs) + 2
        regenerated = False

        if support == "UNSUPPORTED" and len(rel_docs) > 1:
            top_docs = rel_docs[: max(1, len(rel_docs) // 2)]
            gen2 = self._generate(ctx.query, top_docs, language=language)
            answer = (gen2["content"] or "").strip()
            api_calls += 1
            regenerated = True

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
                "fallback_to_all": fallback_to_all,
                "regenerated": regenerated,
                "prompt_tokens": gen.get("prompt_tokens", 0),
                "completion_tokens": gen.get("completion_tokens", 0),
                **ctx.meta,
            },
        )
