"""方向1：自适应矫正器 —— 根据文档特征自动路由到最优方法。

核心逻辑：
1. 快速文档扫描 (1次API调用)：检测是否存在明显矛盾/反事实陈述
2. 路由：no CF detected → confidence (CoT证据链)
         CF detected    → prompt (标注来源+指出矛盾)
         不确定         → voting (多Persona稳健兜底)

总API调用：1(检测) + 1~3(目标方法) = 2~4次
"""
from __future__ import annotations

from ..noise_injector import NoisyContext
from ..prompts import format_context, NAIVE_SYSTEM_ZH
from ..rag_pipeline import RAGResult
from ..correctors.prompt_corrector import PromptCorrector
from ..correctors.confidence_corrector import ConfidenceCorrector
from ..correctors.voting_corrector import EvidenceVotingCorrector
from .base import BaseCorrector, register_corrector


# 噪声检测 prompt：让LLM快速扫描文档中的矛盾
NOISE_DETECT_SYSTEM = (
    "你是文档质量检测器。请快速扫描给定文档，判断其中是否包含"
    "明显的虚构事实、错误编号、或与其他文档直接矛盾的陈述。"
    "只输出一个词：CF（有反事实/矛盾文档）或 CLEAN（文档间一致，无非事实信息）。"
)

NOISE_DETECT_USER = (
    "【问题】{query}\n\n"
    "【文档列表】\n{context}\n\n"
    "请判断这些文档中是否存在反事实/矛盾信息。只输出 CF 或 CLEAN："
)


@register_corrector("adaptive")
class AdaptiveCorrector(BaseCorrector):
    """自适应矫正器：自动检测噪音类型并路由到最优方法。"""

    api_cost = 0  # varies

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        # 懒加载子矫正器
        self._prompt_corr = None
        self._confidence_corr = None
        self._voting_corr = None

    def _detect_noise_type(self, query: str, docs: list[str]) -> str:
        """返回 'CF', 'CLEAN', 或 'UNCERTAIN'。"""
        ctx = format_context(docs, max_chars_per_doc=1000)
        out = self.llm.chat(
            [
                {"role": "system", "content": NOISE_DETECT_SYSTEM},
                {"role": "user", "content": NOISE_DETECT_USER.format(
                    query=query, context=ctx
                )},
            ],
            max_tokens=8,
        )
        content = (out.get("content", "") or "").strip().upper()
        if "CF" in content:
            return "CF"
        elif "CLEAN" in content:
            return "CLEAN"
        else:
            return "UNCERTAIN"

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        # Step 1: 检测噪音类型
        detected = self._detect_noise_type(ctx.query, ctx.docs)

        # Step 2: 路由到最优方法
        if detected == "CF":
            if self._prompt_corr is None:
                self._prompt_corr = PromptCorrector(llm=self.llm)
            result = self._prompt_corr.correct(ctx, language=language)
            route_reason = "CF detected → PromptCorrector (标注来源+指出矛盾)"
        elif detected == "CLEAN":
            if self._confidence_corr is None:
                self._confidence_corr = ConfidenceCorrector(llm=self.llm)
            result = self._confidence_corr.correct(ctx, language=language)
            route_reason = "CLEAN detected → ConfidenceCorrector (CoT证据链)"
        else:
            if self._voting_corr is None:
                self._voting_corr = EvidenceVotingCorrector(llm=self.llm)
            result = self._voting_corr.correct(ctx, language=language)
            route_reason = "UNCERTAIN → VotingCorrector (多Persona稳健兜底)"

        # 记录路由决策
        result.metadata["adaptive_route"] = detected
        result.metadata["adaptive_reason"] = route_reason
        result.metadata["method"] = "adaptive"
        api_calls = 1 + (result.metadata.get("api_calls", 1))
        result.metadata["api_calls"] = api_calls

        return result
