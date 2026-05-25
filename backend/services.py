"""Business logic — HTML rendering, verdict classification."""
from __future__ import annotations

import html as html_mod

from src.data_loader import RGBRecord


# ── Doc card rendering ──

_LABEL_STYLES = {
    "positive":       ("#15803d", "#166534", "POSITIVE"),
    "negative":       ("#b91c1c", "#991b1b", "NEGATIVE"),
    "positive_wrong": ("#b45309", "#92400e", "COUNTERFACTUAL"),
}


def doc_card_html(idx: int, doc: str, label: str, max_chars: int = 500) -> str:
    text = doc[:max_chars] + ("…" if len(doc) > max_chars else "")
    safe = html_mod.escape(text).replace("\n", "<br>")
    border, lc, badge = _LABEL_STYLES.get(label, ("#64748b", "#334155", label.upper()))
    return (
        f"<article class='doc-card' style='border-left-color:{border};'>"
        f"<header class='doc-meta' style='color:{lc};'>文档 [{idx}] · {badge}</header>"
        f"<div class='doc-body'>{safe}</div>"
        f"</article>"
    )


def render_retrieval_html(record: RGBRecord) -> str:
    parts = [
        "<div class='stage-summary'>"
        f"原始检索池 · positive=<b>{len(record.positive)}</b> · "
        f"negative=<b>{len(record.negative)}</b> · "
        f"positive_wrong=<b>{len(record.positive_wrong)}</b></div>"
    ]
    for i, d in enumerate(record.positive):
        parts.append(doc_card_html(i, d, "positive"))
    for i, d in enumerate(record.negative):
        parts.append(doc_card_html(i, d, "negative"))
    for i, d in enumerate(record.positive_wrong):
        parts.append(doc_card_html(i, d, "positive_wrong"))
    return "".join(parts)


def render_injected_html(ctx) -> str:
    pos = ctx.meta.get("positives", 0)
    noise = ctx.meta.get("noises", len(ctx.docs) - pos)
    parts = [
        "<div class='stage-summary'>"
        f"注入后送给 LLM 的 <b>{len(ctx.docs)}</b> 篇 · "
        f"positive=<b>{pos}</b> / noise=<b>{noise}</b> · 实际比例 = "
        f"<b>{ctx.noise_ratio}</b> · 位置策略 = <b>{ctx.noise_position}</b></div>"
    ]
    for i, (d, lab) in enumerate(zip(ctx.docs, ctx.labels)):
        parts.append(doc_card_html(i, d, lab))
    return "".join(parts)


# ── Verdict ──

def verdict(metrics) -> str:
    """根据评测指标判定回答类别。

    阈值依据：
    - contains == 1.0 即视为命中（contains 取值二元）
    - token_f1 或 rouge_l >= 0.6 之间视为部分正确
    - NAR 显著高于 ISR → 噪音主导
    """
    if metrics.contains >= 1.0:
        return "correct"
    if metrics.token_f1 >= 0.6 or metrics.rouge_l >= 0.6:
        return "partial"
    if metrics.nar > metrics.isr + 0.1 and metrics.nar >= 0.3:
        return "noise_biased"
    return "wrong"
