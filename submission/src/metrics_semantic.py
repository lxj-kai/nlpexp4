"""方向2：语义级文档溯源 —— 升级 ISR/NAR。

从"字符级 token 重叠"升级为"LLM驱动的句子级信息归属判定"。

每个答案会按句子拆开，由 LLM 判定每句的信息来源：
- from_positive: 源自 positive 文档
- from_negative: 源自 negative/positive_wrong 文档
- from_neither: 无法归属（可能是推理/总结）
- from_hallucination: 在所有文档中都找不到（幻觉）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm_client import LLMClient, get_client
from .prompts import format_context


ATTR_SYSTEM = (
    "你是信息溯源专家。给定一个问题、若干文档、和模型生成的答案，"
    "请将答案拆分为最小信息单元（每条事实或关键短语），"
    "并标注每个信息单元来源于哪篇文档（文档编号），"
    "或在所有文档中都找不到。\n\n"
    "输出格式（每行一条）：\n"
    "信息单元: <文本> | 来源: [文档i] | 可信度: high/mid/low\n"
    "如果找不到来源，来源写 [无]。\n"
    "如果信息单元明显是文档中不存在的编造内容，可信度标注为 hallucination。"
)

ATTR_USER = (
    "【问题】{query}\n\n"
    "【文档】\n{context}\n\n"
    "【模型答案】{answer}\n\n"
    "请逐条溯源（每条一行，最多输出10条）："
)


@dataclass
class SemanticAttribution:
    """语义级信息来源归因结果。"""

    n_units: int = 0
    units: list[dict] = field(default_factory=list)

    # 聚合统计
    n_from_positive: int = 0
    n_from_negative: int = 0
    n_from_neither: int = 0
    n_hallucination: int = 0

    isr_semantic: float = 0.0  # positive 归属比例
    nar_semantic: float = 0.0  # negative 归属比例

    raw_response: str = ""


def attribute_answer_semantic(
    query: str,
    answer: str,
    docs: list[str],
    labels: list[str],
    *,
    llm: LLMClient | None = None,
) -> SemanticAttribution:
    """LLM驱动的语义级答案溯源。

    返回每个信息单元的文档归属及聚合 ISR/NAR。
    """
    if not answer or not answer.strip():
        return SemanticAttribution()

    llm = llm or get_client()
    ctx = format_context(docs, max_chars_per_doc=1200)

    out = llm.chat(
        [
            {"role": "system", "content": ATTR_SYSTEM},
            {"role": "user", "content": ATTR_USER.format(
                query=query, context=ctx, answer=answer
            )},
        ],
        max_tokens=512,
    )
    raw = (out.get("content") or "").strip()
    if not raw:
        return SemanticAttribution(raw_response=raw)

    # 解析输出
    units = []
    n_pos = n_neg = n_neither = n_hallu = 0

    for line in raw.split("\n"):
        line = line.strip()
        if not line or "信息单元" not in line:
            continue

        # 提取信息单元文本
        text = ""
        if "信息单元:" in line:
            text = line.split("信息单元:", 1)[1].split("|")[0].strip()
        elif "信息单元：" in line:
            text = line.split("信息单元：", 1)[1].split("|")[0].strip()

        if not text:
            continue

        # 判断来源
        source = line.split("来源:", 1)[-1].split("|")[0].strip() if "来源:" in line else ""
        if not source and "来源：" in line:
            source = line.split("来源：", 1)[-1].split("|")[0].strip()

        credibility = ""
        if "可信度:" in line:
            credibility = line.split("可信度:", 1)[-1].strip()
        elif "可信度：" in line:
            credibility = line.split("可信度：", 1)[-1].strip()

        # 分类
        if "hallucination" in credibility.lower():
            category = "hallucination"
            n_hallu += 1
        elif "[无]" in source or "无" == source.strip():
            category = "neither"
            n_neither += 1
        else:
            # 尝试匹配文档索引来确定是 positive 还是 negative
            doc_idx = -1
            for part in source.replace("[", "").replace("]", "").split(","):
                part = part.strip()
                if part.startswith("文档"):
                    try:
                        doc_idx = int(part.replace("文档", "").strip())
                    except ValueError:
                        pass

            if 0 <= doc_idx < len(labels):
                if labels[doc_idx] == "positive":
                    category = "positive"
                    n_pos += 1
                else:
                    category = "negative"
                    n_neg += 1
            else:
                category = "neither"
                n_neither += 1

        units.append({
            "text": text,
            "source": source,
            "credibility": credibility,
            "category": category,
        })

    total = len(units)
    return SemanticAttribution(
        n_units=total,
        units=units,
        n_from_positive=n_pos,
        n_from_negative=n_neg,
        n_from_neither=n_neither,
        n_hallucination=n_hallu,
        isr_semantic=n_pos / total if total > 0 else 0.0,
        nar_semantic=n_neg / total if total > 0 else 0.0,
        raw_response=raw,
    )


def compare_attributions(
    token_attr: Any,   # SourceAttribution from metrics.py
    semantic_attr: SemanticAttribution,
) -> dict:
    """对比字符级和语义级溯源结果的差异。"""
    return {
        "token_isr": getattr(token_attr, "isr", 0),
        "token_nar": getattr(token_attr, "nar", 0),
        "semantic_isr": semantic_attr.isr_semantic,
        "semantic_nar": semantic_attr.nar_semantic,
        "semantic_hallucination": semantic_attr.n_hallucination,
        "semantic_units": semantic_attr.n_units,
        "isr_delta": semantic_attr.isr_semantic - getattr(token_attr, "isr", 0),
        "nar_delta": semantic_attr.nar_semantic - getattr(token_attr, "nar", 0),
    }
