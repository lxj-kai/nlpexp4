"""Gradio Demo —— 交互式展示 RAG 噪音注入与矫正全流程。

用法：
    python demo/app.py
默认 http://127.0.0.1:7861
"""
from __future__ import annotations

import html as _html
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
    safe_text = _html.escape(text).replace("\n", "<br>")
    if label == "positive":
        border_color, label_color = "#15803d", "#166534"
    elif label == "positive_wrong":
        border_color, label_color = "#b45309", "#92400e"
    else:
        border_color, label_color = "#b91c1c", "#991b1b"
    return (
        f"<div class='doc-card' style='border-left:4px solid {border_color};'>"
        f"<div class='doc-meta' style='color:{label_color};'>"
        f"{emoji} <span style='font-variant:small-caps;letter-spacing:.5px;'>文档 [{idx}]</span> "
        f"<span style='opacity:.85;'>·</span> {badge}</div>"
        f"<div class='doc-body'>{safe_text}</div>"
        f"</div>"
    )


def _render_retrieval(record) -> str:
    """渲染原始检索池（注入前）：positive + negative + positive_wrong 全部展开。"""
    parts: list[str] = []
    parts.append(
        f"<div class='stage-summary'>"
        f"原始检索池 · positive=<b>{len(record.positive)}</b> · negative=<b>{len(record.negative)}</b>"
        f" · positive_wrong=<b>{len(record.positive_wrong)}</b></div>"
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
        f"<div class='stage-summary'>"
        f"注入后送给 LLM 的 <b>{len(ctx.docs)}</b> 篇文档 · "
        f"positive=<b>{pos}</b> / noise=<b>{noise}</b> · 实际比例 = "
        f"<b style='color:#b45309'>{ctx.noise_ratio}</b> · "
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
    try:
        if not sample_label:
            return "", "", ""
        record = _find_record(language, subset, sample_label)
        return (
            record.query,
            " / ".join(record.answers_norm) or "(无)",
            _render_retrieval(record),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERR] {e}", "", f"<div style='color:#ef4444'>渲染失败：{e}</div>"


def _on_inject(language, subset, sample_label, noise_ratio, noise_type, noise_position):
    """只跑噪音注入（不调 LLM），用于"先看注入效果"。"""
    if not sample_label:
        return "⚠️ 请先选样本", "", ""
    try:
        record = _find_record(language, subset, sample_label)
        ctx = inject(
            record,
            noise_ratio=noise_ratio,
            noise_type=noise_type,
            noise_position=noise_position,
            max_docs=CONFIG.max_docs,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ 注入失败：{e}", "", ""
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
        return "⚠️ 请先选样本", "", "", "", ""
    try:
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ 推理失败：{e}", "", "", "", ""

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
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&family=Source+Serif+Pro:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
  max-width: 1400px !important;
  font-family: 'Source Serif Pro', 'Noto Serif SC', 'Songti SC', 'STSong', Georgia, serif !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
  font-family: 'Source Serif Pro', 'Noto Serif SC', Georgia, serif !important;
  font-weight: 700;
  letter-spacing: .2px;
  color: #1e293b;
}
.gradio-container h2 {
  border-bottom: 2px solid #cbd5e1;
  padding-bottom: 6px;
  margin-top: 18px;
}
.gradio-container .markdown, .gradio-container p, .gradio-container li {
  font-size: 15px;
  line-height: 1.75;
  color: #1f2937;
}
.gradio-container code, .gradio-container pre {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace !important;
  font-size: 13.5px;
}
.gradio-container button {
  font-family: 'Source Serif Pro', 'Noto Serif SC', serif !important;
  font-weight: 600;
  letter-spacing: .3px;
}
.gradio-container label, .gradio-container .label-wrap {
  font-family: 'Source Serif Pro', 'Noto Serif SC', serif !important;
  font-weight: 600;
  color: #334155;
}

.doc-card {
  padding: 12px 16px;
  margin: 10px 0;
  background: #fafaf7;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.doc-meta {
  font-weight: 700;
  font-size: 13.5px;
  letter-spacing: .3px;
  margin-bottom: 6px;
  font-family: 'Source Serif Pro', 'Noto Serif SC', serif;
}
.doc-body {
  color: #111827;
  line-height: 1.78;
  font-size: 15px;
  font-family: 'Noto Serif SC', 'Source Serif Pro', 'Songti SC', 'STSong', Georgia, serif;
  text-align: justify;
}
.stage-summary {
  font-size: 14px;
  color: #475569;
  margin: 8px 0 10px;
  padding: 10px 14px;
  background: #f1f5f9;
  border-left: 3px solid #475569;
  border-radius: 2px;
  font-family: 'Source Serif Pro', 'Noto Serif SC', serif;
}
.stage-summary b {
  color: #0f172a;
}
table {
  font-family: 'Source Serif Pro', 'Noto Serif SC', serif !important;
  font-size: 14.5px;
}
table th {
  background: #1e293b !important;
  color: #f8fafc !important;
  font-weight: 600;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG 噪音鲁棒性 Demo",
        theme=gr.themes.Default(
            primary_hue="slate",
            neutral_hue="slate",
            font=["Source Serif Pro", "Noto Serif SC", "Songti SC", "Georgia", "serif"],
        ),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# 面向 RAG 检索噪音的鲁棒性推理实验平台\n"
            "**实验流程**：① 选取问题 → ② 查看原始检索池 → ③ 注入噪音并构造上下文 → ④ 审视最终 Prompt → ⑤ 调用大模型推理 → ⑥ 对比答案与指标。"
        )

        # ===== 顶部控制栏 =====
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### ① 数据 & 样本")
                language = gr.Radio(["zh", "en"], value="zh", label="语言")
                subset = gr.Dropdown(
                    ["main", "refine", "fact", "int"], value="main", label="子集"
                )
                _initial_choices = _list_samples("zh", "main")
                sample = gr.Dropdown(
                    label="选择问题",
                    choices=_initial_choices,
                    value=_initial_choices[0] if _initial_choices else None,
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
            return gr.update(choices=choices, value=choices[0] if choices else None)

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
