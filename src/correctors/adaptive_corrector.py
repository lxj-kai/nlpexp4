"""方向1：自适应矫正器 —— 噪音类型检测 + 方法路由。

核心逻辑：
Step 1: 噪音检测器 (1次API, max_tokens=16)
  - 扫描文档是否存在反事实/矛盾陈述
  - 输出 CF / CLEAN / UNCERTAIN
Step 2: 路由到最优方法
  - CF         → PromptCorrector (标注来源+指出矛盾, 反事实下F1=0.781)
  - CLEAN      → ConfidenceCorrector (CoT证据链, 语义下F1=0.791)
  - UNCERTAIN  → VotingCorrector (多Persona稳健兜底)

总API调用：1(检测) + 1~4(目标方法) = 2~5次

设计依据：互锁效应表明最优方法取决于噪音类型。检测器用独立prompt模板
避免与RAG/矫正调用共享缓存key。
"""
from __future__ import annotations

from ..noise_injector import NoisyContext
from ..prompts import format_context
from ..rag_pipeline import RAGResult
from ..correctors.prompt_corrector import PromptCorrector
from ..correctors.confidence_corrector import ConfidenceCorrector
from ..correctors.voting_corrector import EvidenceVotingCorrector
from .base import BaseCorrector, register_corrector


# ── 噪音检测 Prompt（中英双语，带 few-shot 约束） ──

NOISE_DETECT_SYSTEM_ZH = (
    "你是文档质量检测器。你的任务是快速扫描给定文档集合，判断其中是否包含"
    "虚构事实、错误编号、或与其他文档直接矛盾的陈述。\n\n"
    "判断标准：\n"
    "- CF（CounterFactual）：文档中存在明确的反事实/虚构/矛盾信息。例如：同一事实多个文档给出不同版本、包含明显错误的数据或编号。\n"
    "- CLEAN：所有文档之间信息一致，没有虚构或矛盾的内容。\n\n"
    "你必须只输出一个词：CF 或 CLEAN。不要输出任何其他文字。"
)

NOISE_DETECT_SYSTEM_EN = (
    "You are a document quality inspector. Scan the given documents and determine "
    "whether they contain fabricated facts, wrong identifiers, or statements that "
    "directly contradict each other.\n\n"
    "Criteria:\n"
    "- CF (CounterFactual): Documents contain clearly fabricated/contradictory info, "
    "e.g. different docs give conflicting versions of the same fact.\n"
    "- CLEAN: All documents are consistent, no fabricated or contradictory content.\n\n"
    "You MUST output exactly one word: CF or CLEAN. No other text."
)

NOISE_DETECT_USER_ZH = (
    "【问题】{query}\n\n"
    "【文档列表】\n{context}\n\n"
    "请判断这些文档中是否存在反事实或矛盾信息。只输出 CF 或 CLEAN："
)

NOISE_DETECT_USER_EN = (
    "[Question] {query}\n\n"
    "[Documents]\n{context}\n\n"
    "Do these documents contain counterfactual or contradictory information? "
    "Output only CF or CLEAN:"
)


@register_corrector("adaptive")
class AdaptiveCorrector(BaseCorrector):
    """自适应矫正器：噪音检测 + 方法路由。"""

    api_cost = 0  # 运行时动态计算

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self._prompt_corr = None
        self._confidence_corr = None
        self._voting_corr = None
        # 统计路由决策
        self.route_stats: dict[str, int] = {"CF": 0, "CLEAN": 0, "UNCERTAIN": 0}

    def _detect_noise_type(self, query: str, docs: list[str], language: str = "zh") -> str:
        """返回 'CF', 'CLEAN', 或 'UNCERTAIN'。

        使用独立系统提示和 max_tokens=16（与 RAG 调用隔离缓存），
        同时提供足够的 token 余量避免截断。
        """
        is_zh = language == "zh"
        system = NOISE_DETECT_SYSTEM_ZH if is_zh else NOISE_DETECT_SYSTEM_EN
        user_tmpl = NOISE_DETECT_USER_ZH if is_zh else NOISE_DETECT_USER_EN

        ctx = format_context(docs, max_chars_per_doc=1200)
        out = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_tmpl.format(query=query, context=ctx)},
            ],
            max_tokens=16,   # 独立于 512 的 RAG 调用，杜绝缓存碰撞
            temperature=0.0,  # 确定性输出
        )
        content = (out.get("content", "") or "").strip().upper()

        # 多级解析：精确匹配 → 包含匹配 → 中文语义匹配
        if content in ("CF", "CLEAN"):
            return content

        if "CF" in content and "CLEAN" not in content:
            return "CF"
        if "CLEAN" in content and "CF" not in content:
            return "CLEAN"

        # 中文模型可能输出中文
        if any(w in content for w in ("反事实", "有矛盾", "存在矛盾", "虚构", "冲突")):
            return "CF"
        if any(w in content for w in ("清洁", "干净", "无矛盾", "一致", "无虚构")):
            return "CLEAN"

        return "UNCERTAIN"

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        # Step 1: 噪音类型检测
        detected = self._detect_noise_type(ctx.query, ctx.docs, language=language)
        self.route_stats[detected] = self.route_stats.get(detected, 0) + 1

        # Step 2: 路由
        if detected == "CF":
            if self._prompt_corr is None:
                self._prompt_corr = PromptCorrector(llm=self.llm)
            result = self._prompt_corr.correct(ctx, language=language)
            route_reason = f"CF detected → PromptCorrector (标注来源+指出矛盾)"
        elif detected == "CLEAN":
            if self._confidence_corr is None:
                self._confidence_corr = ConfidenceCorrector(llm=self.llm)
            result = self._confidence_corr.correct(ctx, language=language)
            route_reason = f"CLEAN detected → ConfidenceCorrector (CoT证据链)"
        else:
            if self._voting_corr is None:
                self._voting_corr = EvidenceVotingCorrector(llm=self.llm)
            result = self._voting_corr.correct(ctx, language=language)
            route_reason = f"UNCERTAIN → VotingCorrector (多Persona稳健兜底)"

        # 记录元信息
        result.metadata["adaptive_route"] = detected
        result.metadata["adaptive_reason"] = route_reason
        result.metadata["method"] = "adaptive"
        target_api = result.metadata.get("api_calls", 1)
        result.metadata["api_calls"] = 1 + target_api

        return result
