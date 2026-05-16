"""Gradio Demo —— 交互式展示 RAG 问答 + 噪音注入 + 矫正效果。

用法：
    python demo/app.py
默认 http://127.0.0.1:7860
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


def _format_docs(docs: list[str], labels: list[str]) -> str:
    parts: list[str] = []
    for i, (d, lab) in enumerate(zip(docs, labels)):
        emoji = "🟢" if lab == "positive" else ("🟠" if lab == "positive_wrong" else "🔴")
        parts.append(f"{emoji} **[文档{i}] ({lab})**\n{d[:600]}{'...' if len(d)>600 else ''}")
    return "\n\n---\n\n".join(parts)


def _format_metrics(rag_result, evaluator) -> str:
    m = evaluator.evaluate_one(rag_result)
    return (
        f"- **EM**: {m.em:.2f}\n"
        f"- **Contains**: {m.contains:.2f}\n"
        f"- **Token-F1**: {m.token_f1:.3f}\n"
        f"- **ROUGE-L**: {m.rouge_l:.3f}\n"
        f"- **ISR (信息溯源率)**: {m.isr:.3f}\n"
        f"- **NAR (噪音采纳率)**: {m.nar:.3f}"
    )


def _run(language, subset, sample_label, noise_ratio, noise_type, noise_position, method):
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

    docs_md = _format_docs(ctx.docs, ctx.labels)
    metrics_md = _format_metrics(result, _EVAL)
    meta = (
        f"- noise_ratio (actual): **{ctx.noise_ratio}** | type: {ctx.noise_type} | "
        f"position: {ctx.noise_position}\n"
        f"- docs: {len(ctx.docs)} (positive={ctx.meta.get('positives',0)})\n"
        f"- method: **{method}** | api_calls={result.metadata.get('api_calls','?')}"
    )
    return record.query, " / ".join(record.answers_norm), result.prediction, metrics_md, meta, docs_md


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG 噪音鲁棒性 Demo",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(
            "# RAG 噪音鲁棒性推理 Demo\n"
            "调节左侧的噪音变量与矫正方法，对比同一题在不同条件下的答案、指标与文档命中。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Radio(["zh", "en"], value="zh", label="语言")
                subset = gr.Dropdown(
                    ["main", "refine", "fact", "int"], value="main", label="子集"
                )
                sample = gr.Dropdown(label="选择问题", choices=_list_samples("zh", "main"))

                noise_ratio = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="噪音比例")
                noise_type = gr.Radio(
                    ["semantic", "counterfactual", "mixed"],
                    value="semantic",
                    label="噪音类型",
                )
                noise_position = gr.Radio(
                    ["front", "back", "interleave", "surround"],
                    value="interleave",
                    label="噪音位置",
                )
                method = gr.Dropdown(
                    ["naive", *list_correctors()], value="naive", label="生成 / 矫正方法"
                )
                run_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                query_out = gr.Textbox(label="问题", interactive=False)
                gold_out = gr.Textbox(label="参考答案", interactive=False)
                pred_out = gr.Textbox(label="模型答案", interactive=False, lines=2)
                meta_out = gr.Markdown(label="运行配置")
                metrics_out = gr.Markdown(label="评估指标")
                with gr.Accordion("检索文档（🟢positive / 🔴negative / 🟠positive_wrong）", open=False):
                    docs_out = gr.Markdown()

        def _refresh_samples(lang, sub):
            choices = _list_samples(lang, sub)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        language.change(_refresh_samples, [language, subset], [sample])
        subset.change(_refresh_samples, [language, subset], [sample])

        run_btn.click(
            _run,
            [language, subset, sample, noise_ratio, noise_type, noise_position, method],
            [query_out, gold_out, pred_out, metrics_out, meta_out, docs_out],
        )

    return demo


def main() -> None:
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
