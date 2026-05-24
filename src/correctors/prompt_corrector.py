"""方法 A — Prompt-aware correction.

通过在 system prompt 中显式引导模型质疑文档，
不增加额外 API 调用次数。
"""
from __future__ import annotations

from ..noise_injector import NoisyContext
from ..prompts import PROMPT_AWARE_SYSTEM_EN, PROMPT_AWARE_SYSTEM_ZH, NAIVE_USER_TMPL, format_context
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


@register_corrector("prompt")
class PromptCorrector(BaseCorrector):
    """方法 A：仅修改 system prompt，让模型主动识别并跳过噪音。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        system = PROMPT_AWARE_SYSTEM_ZH if language == "zh" else PROMPT_AWARE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(
            query=ctx.query, n=len(ctx.docs), context=format_context(ctx.docs)
        )
        out = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=out["content"].strip(),
            docs=ctx.docs,
            labels=list(ctx.labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "latency": out.get("latency", 0.0),
                "api_calls": 1,
                **ctx.meta,
            },
        )
