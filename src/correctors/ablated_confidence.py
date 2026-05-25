"""方向3：Confidence消融变体 + 证据链可视化。

四个变体：
- C_full: 完整4步 (拆解需求 → 证据匹配 → 综合 → 标签输出)
- C_no_decompose: 跳过Step1"拆解信息需求"
- C_no_evidence: 跳过Step2"逐篇证据匹配" → 退化为普通CoT
- C_no_tag: 去掉<answer>标签约束 → 自由格式输出

每个变体记录完整的中间推理过程用于可视化。
"""
from __future__ import annotations

import re
import time

from ..noise_injector import NoisyContext
from ..prompts import NAIVE_USER_TMPL, format_context
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector


_ANS_PAT = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
_FALLBACK_PAT = re.compile(r"最终答案[:：]\s*(.+)")
_FALLBACK2_PAT = re.compile(r"Step4[.。]\s*(.+)")
_FALLBACK3_PAT = re.compile(r"综合证据[:：]\s*(.+)")


def _extract_answer(raw: str) -> str:
    if not raw:
        return ""
    for pat in [_ANS_PAT, _FALLBACK_PAT, _FALLBACK2_PAT, _FALLBACK3_PAT]:
        m = pat.search(raw)
        if m:
            return m.group(1).strip()
    # last resort: last non-empty line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _extract_evidence_section(raw: str) -> str:
    """提取证据匹配段落用于可视化。"""
    lines = raw.split("\n")
    evidence_lines = []
    in_evidence = False
    for line in lines:
        if "证据匹配" in line or "Step2" in line:
            in_evidence = True
        elif "Step3" in line or "综合证据" in line or "最终答案" in line or "<answer>" in line:
            in_evidence = False
        elif in_evidence and line.strip():
            evidence_lines.append(line.strip())
    return "\n".join(evidence_lines)


# ===== Prompt Variants =====

# C_no_decompose: 跳过信息需求拆解
C_NO_DECOMPOSE_SYSTEM = (
    "你是一个证据链推理助手。请严格遵循以下步骤：\n"
    "Step1. 逐一检查文档：标注哪些文档提供了与问题直接相关的信息；\n"
    "Step2. 综合证据：仅基于已找到证据的信息点构建答案；\n"
    "Step3. 输出最终答案（一行，简短）。\n\n"
    "请用如下结构输出：\n"
    "证据匹配：文档[i]→相关信息；...\n"
    "最终答案：<answer>...</answer>"
)

# C_no_evidence: 跳过逐篇核对 → 退化为普通CoT
C_NO_EVIDENCE_SYSTEM = (
    "你是一个推理助手。请回答问题：\n"
    "Step1. 拆解问题：列出回答该问题需要哪些关键信息点；\n"
    "Step2. 思考：从文档中找到了哪些相关证据；\n"
    "Step3. 给出最终答案（一行，简短）。\n\n"
    "请用如下结构输出：\n"
    "信息需求：...\n"
    "相关证据：...\n"
    "最终答案：<answer>...</answer>"
)

# C_no_tag: 去掉 <answer> 标签约束
C_NO_TAG_SYSTEM = (
    "你是一个证据链推理助手。请严格遵循以下步骤：\n"
    "Step1. 拆解问题：列出回答该问题需要哪些关键信息点；\n"
    "Step2. 逐一检查文档：标注哪些文档提供了哪些信息点；\n"
    "Step3. 综合证据：仅基于已找到证据的信息点构建答案；\n"
    "Step4. 输出最终答案（一行，简短）。"
)

# C_full (same as original but reproducible here for ablation comparison)
C_FULL_SYSTEM = (
    "你是一个证据链推理助手。请严格遵循以下步骤：\n"
    "Step1. 拆解问题：列出回答该问题需要哪些关键信息点；\n"
    "Step2. 逐一检查文档：标注哪些文档提供了哪些信息点；\n"
    "Step3. 综合证据：仅基于已找到证据的信息点构建答案；\n"
    "Step4. 输出最终答案（一行，简短）。\n\n"
    "请用如下结构输出：\n"
    "信息需求：...\n"
    "证据匹配：文档[i]→信息点x；...\n"
    "最终答案：<answer>...</answer>"
)


def _run_ablated(
    llm,
    ctx: NoisyContext,
    system_prompt: str,
    variant_name: str,
    *,
    language: str = "zh",
) -> RAGResult:
    user = NAIVE_USER_TMPL.format(
        query=ctx.query, n=len(ctx.docs), context=format_context(ctx.docs)
    )
    t0 = time.perf_counter()
    out = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        max_tokens=768,
    )
    raw = out["content"] or ""
    answer = _extract_answer(raw)
    evidence = _extract_evidence_section(raw)
    elapsed = time.perf_counter() - t0

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
            "method": variant_name,
            "variant": variant_name,
            "reasoning": raw,
            "evidence_section": evidence,
            "api_calls": 1,
            "prompt_tokens": out.get("prompt_tokens", 0),
            "completion_tokens": out.get("completion_tokens", 0),
            "latency": elapsed,
            **ctx.meta,
        },
    )


@register_corrector("ablated_full")
class AblatedFullCorrector(BaseCorrector):
    """消融对照：完整4步 (C_full)。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        return _run_ablated(self.llm, ctx, C_FULL_SYSTEM, "C_full", language=language)


@register_corrector("ablated_no_decompose")
class AblatedNoDecomposeCorrector(BaseCorrector):
    """消融对照：去掉信息需求拆解 (C_no_decompose)。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        return _run_ablated(self.llm, ctx, C_NO_DECOMPOSE_SYSTEM, "C_no_decompose", language=language)


@register_corrector("ablated_no_evidence")
class AblatedNoEvidenceCorrector(BaseCorrector):
    """消融对照：去掉逐篇证据匹配 (C_no_evidence) —— 最关键的消融。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        return _run_ablated(self.llm, ctx, C_NO_EVIDENCE_SYSTEM, "C_no_evidence", language=language)


@register_corrector("ablated_no_tag")
class AblatedNoTagCorrector(BaseCorrector):
    """消融对照：去掉结构标签约束 (C_no_tag)。"""

    api_cost = 1

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        return _run_ablated(self.llm, ctx, C_NO_TAG_SYSTEM, "C_no_tag", language=language)
