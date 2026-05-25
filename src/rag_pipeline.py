"""Naive RAG pipeline —— 直接拼接文档送入 LLM。

这是所有矫正方法的 baseline。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from tqdm import tqdm

from .config import CONFIG
from .llm_client import LLMClient, get_client
from .noise_injector import NoisyContext
from .prompts import build_naive_prompt
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """单条样本的端到端生成结果。"""

    sample_id: int
    query: str
    gold_answers: list[str]
    prediction: str
    docs: list[str]
    labels: list[str]
    noise_ratio: float
    noise_type: str
    noise_position: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """Naive RAG：把所有文档拼接 → 调用 LLM → 返回答案。"""

    method_name: str = "naive"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_client()

    def answer(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        system, user = build_naive_prompt(ctx.query, ctx.docs, language=language)
        out = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
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
                "method": self.method_name,
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "latency": out.get("latency", 0.0),
                "cached": out.get("cached", False),
                **ctx.meta,
            },
        )

    def batch_answer(
        self,
        contexts: list[NoisyContext],
        *,
        language: str = "zh",
        show_progress: bool = True,
    ) -> list[RAGResult]:
        iterator = tqdm(
            contexts, desc=f"RAG/{self.method_name}", disable=not show_progress
        )
        results: list[RAGResult] = []
        for ctx in iterator:
            try:
                results.append(self.answer(ctx, language=language))
            except Exception as e:
                logger.exception(f"sample {ctx.sample_id} failed: {e}")
        return results
