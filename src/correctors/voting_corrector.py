"""方法 D — Evidence Voting (Persona Ensemble + LLM Aggregator).

核心思想：用 3 个不同 Persona 的 system prompt 独立推理 → 再让一个 LLM
聚合者投票/合成最终答案。理论上对反事实噪音和模型偶发幻觉更鲁棒。

API 调用：3 (候选) + 1 (聚合) = 4×
"""
from __future__ import annotations

from collections import Counter

from ..noise_injector import NoisyContext
from ..prompts import (
    NAIVE_USER_TMPL,
    VOTE_AGGREGATE_SYSTEM_ZH,
    VOTE_AGGREGATE_USER_TMPL,
    VOTE_PROMPTS_ZH,
    format_context,
)
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


def _majority_or_first(answers: list[str]) -> str | None:
    """若 ≥2 个候选答案规范化后一致，跳过 LLM 聚合直接返回多数派。"""
    if not answers:
        return None
    norm = [a.strip().lower() for a in answers if a and a.strip()]
    if not norm:
        return None
    counter = Counter(norm)
    most_common, count = counter.most_common(1)[0]
    if count >= 2:
        for original in answers:
            if original.strip().lower() == most_common:
                return original
    return None


@register_corrector("voting")
class EvidenceVotingCorrector(BaseCorrector):
    """方法 D：多 Persona 候选 + LLM 聚合。"""

    api_cost = 4

    def _candidate(self, system: str, query: str, docs: list[str]) -> dict:
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
        return self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=256,
        )

    def _aggregate(self, query: str, candidates: list[str]) -> dict:
        return self.llm.chat(
            [
                {"role": "system", "content": VOTE_AGGREGATE_SYSTEM_ZH},
                {
                    "role": "user",
                    "content": VOTE_AGGREGATE_USER_TMPL.format(
                        query=query,
                        cand1=candidates[0],
                        cand2=candidates[1],
                        cand3=candidates[2],
                    ),
                },
            ],
            max_tokens=128,
        )

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        cand_outs = [self._candidate(sys_msg, ctx.query, ctx.docs) for sys_msg in VOTE_PROMPTS_ZH]
        candidates = [(o["content"] or "").strip() for o in cand_outs]

        skipped_agg = _majority_or_first(candidates)
        if skipped_agg is not None:
            final = skipped_agg
            api_calls = len(VOTE_PROMPTS_ZH)
            agg_meta: dict = {"voted_by": "majority"}
        else:
            agg = self._aggregate(ctx.query, candidates)
            final = (agg["content"] or "").strip()
            api_calls = len(VOTE_PROMPTS_ZH) + 1
            agg_meta = {
                "voted_by": "llm_aggregator",
                "agg_tokens": agg.get("completion_tokens", 0),
            }

        prompt_tokens = sum(o.get("prompt_tokens", 0) for o in cand_outs)
        completion_tokens = sum(o.get("completion_tokens", 0) for o in cand_outs)

        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=final,
            docs=ctx.docs,
            labels=list(ctx.labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "candidates": candidates,
                "api_calls": api_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                **agg_meta,
                **ctx.meta,
            },
        )
