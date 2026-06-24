"""矫正方法抽象基类 + 注册表。

所有矫正器只需继承 BaseCorrector 并实现 `correct()`，
实验脚本通过 `get_corrector(name)` 统一调用，无需 if/else 分支。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..llm_client import LLMClient, get_client
from ..noise_injector import NoisyContext
from ..rag_pipeline import RAGResult


from ..utils import get_logger, parallel_map


class BaseCorrector(ABC):
    """矫正方法的统一接口。

    Subclasses MUST set `name` (str) and implement `correct(ctx)`.
    """

    name: str = "base"
    api_cost: int = 1

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or get_client()

    @abstractmethod
    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        """对带噪音上下文进行矫正生成。"""
        raise NotImplementedError

    def batch_correct(
        self,
        contexts: list[NoisyContext],
        *,
        language: str = "zh",
        show_progress: bool = True,
        workers: int = 1,
    ) -> list[RAGResult]:
        log = get_logger(__name__)

        def _one(ctx: NoisyContext) -> RAGResult | None:
            try:
                return self.correct(ctx, language=language)
            except Exception as e:
                log.exception(f"sample {ctx.sample_id} {self.name} failed: {e}")
                return None

        out = parallel_map(
            contexts,
            _one,
            workers=workers,
            show_progress=show_progress,
            desc=f"correct/{self.name}",
        )
        return [r for r in out if r is not None]


_REGISTRY: dict[str, type[BaseCorrector]] = {}


def register_corrector(name: str) -> Callable[[type[BaseCorrector]], type[BaseCorrector]]:
    """装饰器：把矫正方法类登记到全局注册表。"""

    def _wrap(cls: type[BaseCorrector]) -> type[BaseCorrector]:
        if name in _REGISTRY:
            raise ValueError(f"corrector {name} 已注册")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_corrector(name: str, **kwargs) -> BaseCorrector:
    if name not in _REGISTRY:
        raise KeyError(f"未知 corrector: {name}; 可选: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_correctors() -> list[str]:
    return sorted(_REGISTRY.keys())
