"""方法 C — CoT-Evidence reasoning.

强制 LLM 在回答前显式拆解信息需求、逐篇核对文档证据、
最后用 <answer>...</answer> 标签输出答案。

只 1 次 API 调用，但 prompt 更长。
"""
from __future__ import annotations

import re

from ..noise_injector import NoisyContext
from ..prompts import COT_EVIDENCE_SYSTEM_ZH, NAIVE_USER_TMPL, format_context
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


_ANS_PAT = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
_FALLBACK_PAT = re.compile(r"最终答案[:：]\s*(.+)")


def _extract_answer(raw: str) -> str:
    if not raw:
        return ""
    m = _ANS_PAT.search(raw)
    if m:
        return m.group(1).strip()
    m = _FALLBACK_PAT.search(raw)
    if m:
        return m.group(1).strip().splitlines()[0]
    return raw.strip().splitlines()[-1].strip()


@register_corrector("confidence")
class ConfidenceCorrector(BaseCorrector):
    """方法 C：CoT 证据链推理。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        user = NAIVE_USER_TMPL.format(
            query=ctx.query, n=len(ctx.docs), context=format_context(ctx.docs)
        )
        out = self.llm.chat(
            [
                {"role": "system", "content": COT_EVIDENCE_SYSTEM_ZH},
                {"role": "user", "content": user},
            ],
            max_tokens=768,
        )
        raw = out["content"] or ""
        answer = _extract_answer(raw)
        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=answer,
            docs=ctx.docs,
            labels=list(ctx.labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "reasoning": raw,
                "api_calls": 1,
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "latency": out.get("latency", 0.0),
                **ctx.meta,
            },
        )
