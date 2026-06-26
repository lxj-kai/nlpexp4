"""Context-QA pipeline —— 拼接 RAG 检索上下文（含标注噪音）送入 LLM。

实验设定（题目8）：
  数据集已标注 supporting docs (positive) 与 noisy docs (negative / positive_wrong)，
  分别对应「检索结果中逻辑上可依赖的文档」与「语义相关但缺少逻辑依赖的噪音文档」。
  noise_injector 按设定比例混合后拼接，模拟 RAG 返回的 context window；
  本模块不做向量检索——检索噪声的来源由数据集标注 + 注入比例控制。

这是所有矫正方法的 baseline。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .config import CONFIG
from .llm_client import LLMClient, get_client
from .noise_injector import NoisyContext
from .prompts import build_closed_book_prompt, build_naive_prompt, context_dataset
from .utils import get_logger, parallel_map

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


class ClosedBookPipeline:
    """Closed-book baseline：只给问题，不提供任何检索文档。"""

    method_name: str = "closed_book"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_client()

    def answer(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        system, user = build_closed_book_prompt(ctx.query, language=language)
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
            docs=[],
            labels=[],
            noise_ratio=0.0,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.method_name,
                "closed_book": True,
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
        workers: int = 1,
    ) -> list[RAGResult]:
        def _one(ctx: NoisyContext) -> RAGResult | None:
            try:
                return self.answer(ctx, language=language)
            except Exception as e:
                logger.exception(f"sample {ctx.sample_id} failed: {e}")
                return None

        out = parallel_map(
            contexts,
            _one,
            workers=workers,
            show_progress=show_progress,
            desc=f"RAG/{self.method_name}",
        )
        return [r for r in out if r is not None]


class RAGPipeline:
    """Naive RAG：把所有文档拼接 → 调用 LLM → 返回答案。"""

    method_name: str = "naive"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_client()

    def answer(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        system, user = build_naive_prompt(
            ctx.query, ctx.docs, language=language, dataset=context_dataset(ctx)
        )
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
        workers: int = 1,
    ) -> list[RAGResult]:
        def _one(ctx: NoisyContext) -> RAGResult | None:
            try:
                return self.answer(ctx, language=language)
            except Exception as e:
                logger.exception(f"sample {ctx.sample_id} failed: {e}")
                return None

        out = parallel_map(
            contexts,
            _one,
            workers=workers,
            show_progress=show_progress,
            desc=f"RAG/{self.method_name}",
        )
        return [r for r in out if r is not None]
