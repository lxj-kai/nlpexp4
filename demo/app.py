"""Gradio Demo —— 交互式展示 RAG 噪音注入与矫正全流程。

用法：
    python demo/app.py
默认 http://127.0.0.1:7861
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.config import CONFIG
from src.correctors import get_corrector, list_correctors
from src.data_loader import load_dataset
from src.evaluator import Evaluator
from src.llm_client import LLMClient
from src.noise_injector import inject
from src.prompts import build_naive_prompt, format_context
from src.rag_pipeline import RAGPipeline


CONFIG.ensure_dirs()
_LLM = LLMClient()
_EVAL = Evaluator(use_llm_judge=False)
_RECORDS_CACHE: dict[tuple[str, str], list] = {}


def _load(language: str, subset: str):
    key = (language, subset)
    if key not in _RECORDS_CACHE:
        _RECORDS_CACHE[key] = load_dataset(language=language, subset=subset)
    return _RECORDS_CACHE[key]


def _list_samples(language: str, subset: str):
    recs = _load(language, subset)
    return [f"#{r.id} | {r.query[:40]}" for r in recs]


def _find_record(language: str, subset: str, sample_label: str):
    recs = _load(language, subset)
    try:
        idx = int(sample_label.split("|")[0].strip().lstrip("#"))
    except Exception:
        idx = 0
    for r in recs:
        if r.id == idx:
            return r
    return recs[0]


# ─── 渲染辅助 ───

LABEL_BADGE = {
    "positive": ("🟢", "POSITIVE"),
    "negative": ("🔴", "NEGATIVE"),
    "positive_wrong": ("🟠", "COUNTERFACTUAL"),
}


def _doc_card(idx: int, doc: str, label: str, *, max_chars: int = 600) -> str:
    emoji, badge = LABEL_BADGE.get(label, ("⚪", label.upper()))
    text = doc[:max_chars] + ("…" if len(doc) > max_chars else "")
    return (
        f"<div style='padding:10px 12px;margin:8px 0;border-left:4px solid "
        f"{'#10b981' if label=='positive' else ('#f59e0b' if label=='positive_wrong' else '#ef4444')};"
        f"background:rgba(255,255,255,0.03);border-radius:6px;'>"
        f"<div style='font-weight:600;color:#94a3b8;font-size:.85rem;margin-bottom:4px;'>"
        f"{emoji} 文档 [{idx}] · {badge}</div>"
        f"<div style='color:#e2e8f0;line-height:1.55;font-size:.92rem;'>{gr.utils.sanitize_value(text) if hasattr(gr,'utils') else text}</div>"
        f"</div>"
    )


def _render_retrieval(record) -> str:
    """渲染原始检索池（注入前）：positive + negative + positive_wrong 全部展开。"""
    parts: list[str] = []
    parts.append(
        f"<div style='font-size:.95rem;color:#94a3b8;margin-bottom:6px;'>"
        f"原始检索池 · positive={len(record.positive)} · negative={len(record.negative)}"
        f" · positive_wrong={len(record.positive_wrong)}</div>"
    )
    for i, d in enumerate(record.positive):
        parts.append(_doc_card(i, d, "positive", max_chars=400))
    for i, d in enumerate(record.negative):
        parts.append(_doc_card(i, d, "negative", max_chars=400))
    for i, d in enumerate(record.positive_wrong):
        parts.append(_doc_card(i, d, "positive_wrong", max_chars=400))
    return "\n".join(parts)


def _render_injected(ctx) -> str:
    """渲染噪音注入后送进 LLM 的文档序列。"""
    parts: list[str] = []
    pos = ctx.meta.get("positives", 0)
    noise = ctx.meta.get("noises", len(ctx.docs) - pos)
    parts.append(
        f"<div style='font-size:.95rem;color:#94a3b8;margin-bottom:6px;'>"
        f"注入后送给 LLM 的 <b style='color:#3b82f6'>{len(ctx.docs)}</b> 篇文档 · "
        f"positive={pos} / noise={noise} · 实际比例 = "
        f"<b style='color:#f59e0b'>{ctx.noise_ratio}</b> · "
        f"位置策略 = <b>{ctx.noise_position}</b></div>"
    )
    for i, (d, lab) in enumerate(zip(ctx.docs, ctx.labels)):
        parts.append(_doc_card(i, d, lab))
    return "\n".join(parts)


def _render_prompt(ctx, language: str) -> str:
    """渲染最终送进 LLM 的 prompt（System + User）。"""
    sys_msg, user_msg = build_naive_prompt(ctx.query, ctx.docs, language=language)
    return (
        "**[System]**\n```text\n"
        + sys_msg
        + "\n```\n\n**[User]**\n```text\n"
        + user_msg
        + "\n```"
    )


def _format_metrics(rag_result, evaluator) -> str:
    m = evaluator.evaluate_one(rag_result)
    return (
        "| 指标 | 值 | 说明 |\n"
        "|---|---|---|\n"
        f"| EM | {m.em:.2f} | 完全匹配 |\n"
        f"| Contains | {m.contains:.2f} | 包含匹配（更宽容） |\n"
        f"| Token-F1 | {m.token_f1:.3f} | 词级 F1 |\n"
        f"| ROUGE-L | {m.rouge_l:.3f} | 字符级 LCS |\n"
        f"| **ISR** | **{m.isr:.3f}** | 信息溯源率：答案 token 来自 positive 文档的比例 |\n"
        f"| **NAR** | **{m.nar:.3f}** | 噪音采纳率：答案 token 来自 noise 文档的比例 |\n"
    )


def _verdict(rag_result, evaluator) -> str:
    m = evaluator.evaluate_one(rag_result)
    if m.contains >= 0.9:
        return "<span style='color:#10b981;font-weight:700'>✅ 答对了</span>"
    if m.token_f1 >= 0.5:
        return "<span style='color:#f59e0b;font-weight:700'>⚠️ 部分对</span>"
    if m.nar > m.isr and m.nar > 0.3:
        return "<span style='color:#ef4444;font-weight:700'>❌ 被噪音带偏</span>"
    return "<span style='color:#ef4444;font-weight:700'>❌ 答错</span>"


# ─── 主交互函数 ───

def _on_select_sample(language, subset, sample_label):
    """选完样本立刻显示问题/参考答案/原始检索池。"""
    if not sample_label:
        return "", "", ""
    record = _find_record(language, subset, sample_label)
    return (
        record.query,
        " / ".join(record.answers_norm) or "(无)",
        _render_retrieval(record),
    )


def _on_inject(language, subset, sample_label, noise_ratio, noise_type, noise_position):
    """只跑噪音注入（不调 LLM），用于"先看注入效果"。"""
    if not sample_label:
        return "请先选样本", "", ""
    record = _find_record(language, subset, sample_label)
    try:
        ctx = inject(
            record,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            noise_position=noise_position,
            max_docs=CONFIG.max_docs,
        )
    except ValueError as e:
        return f"❌ {e}", "", ""
    summary = (
        f"### 噪音注入完成\n"
        f"- 文档总数：**{len(ctx.docs)}**\n"
        f"- positive：**{ctx.meta.get('positives',0)}** 篇\n"
        f"- noise：**{ctx.meta.get('noises',0)}** 篇\n"
        f"- 实际噪音比例：**{ctx.noise_ratio}**\n"
        f"- 噪音类型：**{ctx.noise_type}**\n"
        f"- 位置策略：**{ctx.noise_position}**\n"
    )
    return summary, _render_injected(ctx), _render_prompt(ctx, language)


def _on_run(
    language, subset, sample_label, noise_ratio, noise_type, noise_position, method
):
    """完整跑：注入 → 调 LLM → 评估。"""
    if not sample_label:
        return "请先选样本", "", "", "", "", ""
    record = _find_record(language, subset, sample_label)
    ctx = inject(
        record,
        noise_ratio=noise_ratio,
        noise_type=noise_type,
        noise_position=noise_position,
        max_docs=CONFIG.max_docs,
    )

    if method == "naive":
        rag = RAGPipeline(llm=_LLM)
        result = rag.answer(ctx, language=language)
    else:
        corr = get_corrector(method, llm=_LLM)
        result = corr.correct(ctx, language=language)

    inject_summary = (
        f"### 注入摘要\n"
        f"- 文档总数：**{len(ctx.docs)}**（positive **{ctx.meta.get('positives',0)}** + noise **{ctx.meta.get('noises',0)}**）\n"
        f"- 实际比例：**{ctx.noise_ratio}** · 类型：**{ctx.noise_type}** · 位置：**{ctx.noise_position}**\n"
    )
    answer_block = (
        f"### 模型答案\n"
        f"> {result.prediction}\n\n"
        f"**参考答案**：{' / '.join(record.answers_norm) or '(无)'}\n\n"
        f"**判定**：{_verdict(result, _EVAL)}\n\n"
        f"**方法**：`{method}` · "
        f"prompt_tokens={result.metadata.get('prompt_tokens',0)} · "
        f"completion_tokens={result.metadata.get('completion_tokens',0)} · "
        f"latency={result.metadata.get('latency',0.0):.2f}s · "
        f"cached={result.metadata.get('cached',False)}"
    )
    metrics_md = _format_metrics(result, _EVAL)
    return (
        inject_summary,
        _render_injected(ctx),
        _render_prompt(ctx, language),
        answer_block,
        metrics_md,
    )


# ─── UI ───

CSS = """
.gradio-container {max-width: 1400px !important;}
#stages {gap: 6px;}
.stage-card {padding: 12px 16px; border-radius: 10px; background: rgba(59,130,246,.06); border: 1px solid rgba(59,130,246,.2);}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG 噪音鲁棒性 Demo",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# RAG 噪音鲁棒性推理 Demo\n"
            "**完整流程**：① 选问题 → ② 看原始检索池 → ③ 注入噪音查看送 LLM 的文档 → ④ 看最终 prompt → ⑤ LLM 推理 → ⑥ 对比答案 + 指标。"
        )

        # ===== 顶部控制栏 =====
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### ① 数据 & 样本")
                language = gr.Radio(["zh", "en"], value="zh", label="语言")
                subset = gr.Dropdown(
                    ["main", "refine", "fact", "int"], value="main", label="子集"
                )
                sample = gr.Dropdown(
                    label="选择问题",
                    choices=_list_samples("zh", "main"),
                    value=None,
                )
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### ② 噪音参数")
                noise_ratio = gr.Slider(
                    0.0, 1.0, 0.5, step=0.05, label="噪音比例 (negative+CF / 总文档)"
                )
                noise_type = gr.Radio(
                    ["semantic", "counterfactual", "mixed"],
                    value="semantic",
                    label="噪音类型",
                    info="semantic=语义相关但不含答案 / counterfactual=错误事实 / mixed=两者混合",
                )
                noise_position = gr.Radio(
                    ["front", "back", "interleave", "surround"],
                    value="interleave",
                    label="噪音位置",
                )
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### ③ 推理方法")
                method = gr.Dropdown(
                    ["naive", *list_correctors()],
                    value="naive",
                    label="生成 / 矫正方法",
                    info="naive=直接拼接送 LLM；其它=4 类矫正策略",
                )
                with gr.Row():
                    inject_btn = gr.Button("仅注入噪音（不调 LLM）", variant="secondary")
                    run_btn = gr.Button("完整跑（注入 + LLM 推理）", variant="primary")

        # ===== 题目 & 参考 =====
        with gr.Row():
            with gr.Column(scale=2):
                query_out = gr.Textbox(label="问题", interactive=False, lines=2)
            with gr.Column(scale=1):
                gold_out = gr.Textbox(label="参考答案 (gold)", interactive=False)

        gr.Markdown("---")

        # ===== Stage A · 原始检索池 =====
        gr.Markdown("## 🔍 Stage A · 原始检索池（注入前）")
        gr.Markdown(
            "数据集自带的真实检索结果，分为 🟢 positive（含答案）/ 🔴 negative（语义相关但不含答案）/ 🟠 positive_wrong（反事实诱导）。"
        )
        retrieval_out = gr.HTML(label=None)

        # ===== Stage B · 注入后 =====
        gr.Markdown("## 💉 Stage B · 噪音注入后送进 LLM 的文档")
        inject_summary_out = gr.Markdown()
        injected_out = gr.HTML(label=None)

        # ===== Stage C · Prompt =====
        with gr.Accordion("📜 Stage C · 最终送给 LLM 的 Prompt（System + User）", open=False):
            prompt_out = gr.Markdown()

        # ===== Stage D · 推理结果 =====
        gr.Markdown("## 🤖 Stage D · LLM 推理结果")
        answer_out = gr.Markdown()
        metrics_out = gr.Markdown()

        # ─── 事件绑定 ───
        def _refresh_samples(lang, sub):
            choices = _list_samples(lang, sub)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        language.change(_refresh_samples, [language, subset], [sample]).then(
            _on_select_sample,
            [language, subset, sample],
            [query_out, gold_out, retrieval_out],
        )
        subset.change(_refresh_samples, [language, subset], [sample]).then(
            _on_select_sample,
            [language, subset, sample],
            [query_out, gold_out, retrieval_out],
        )
        sample.change(
            _on_select_sample,
            [language, subset, sample],
            [query_out, gold_out, retrieval_out],
        )

        inject_btn.click(
            _on_inject,
            [language, subset, sample, noise_ratio, noise_type, noise_position],
            [inject_summary_out, injected_out, prompt_out],
        )

        run_btn.click(
            _on_run,
            [
                language,
                subset,
                sample,
                noise_ratio,
                noise_type,
                noise_position,
                method,
            ],
            [
                inject_summary_out,
                injected_out,
                prompt_out,
                answer_out,
                metrics_out,
            ],
        )

    return demo


def main() -> None:
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
    )


if __name__ == "__main__":
    main()
