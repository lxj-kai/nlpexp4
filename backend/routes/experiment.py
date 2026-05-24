"""Noise-injection & full-pipeline run endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.deps import evaluator, find_record, llm
from backend.models import InjectRequest, InjectResponse, MetricsOut, RunRequest, RunResponse
from backend.services import render_injected_html, verdict
from src.config import CONFIG
from src.correctors import get_corrector
from src.noise_injector import inject
from src.prompts import build_naive_prompt
from src.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["experiment"])


@router.post("/inject", response_model=InjectResponse)
def api_inject(req: InjectRequest):
    record = find_record(req.language, req.subset, req.sample_id)
    try:
        ctx = inject(
            record,
            noise_ratio=req.noise_ratio,
            noise_type=req.noise_type,
            noise_position=req.noise_position,
            max_docs=CONFIG.max_docs,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    summary = (
        f"### 噪音注入完成\n"
        f"- 文档总数：**{len(ctx.docs)}**\n"
        f"- positive：**{ctx.meta.get('positives', 0)}** 篇\n"
        f"- noise：**{ctx.meta.get('noises', 0)}** 篇\n"
        f"- 实际噪音比例：**{ctx.noise_ratio}**\n"
        f"- 类型：**{ctx.noise_type}** · 位置：**{ctx.noise_position}**"
    )
    sys_msg, user_msg = build_naive_prompt(ctx.query, ctx.docs, language=req.language)
    prompt_md = f"**[System]**\n```\n{sys_msg}\n```\n\n**[User]**\n```\n{user_msg}\n```"
    return InjectResponse(
        summary=summary,
        injected_html=render_injected_html(ctx),
        prompt_markdown=prompt_md,
    )


@router.post("/run", response_model=RunResponse)
def api_run(req: RunRequest):
    record = find_record(req.language, req.subset, req.sample_id)
    try:
        ctx = inject(
            record,
            noise_ratio=req.noise_ratio,
            noise_type=req.noise_type,
            noise_position=req.noise_position,
            max_docs=CONFIG.max_docs,
        )
        if req.method == "naive":
            rag = RAGPipeline(llm=llm)
            result = rag.answer(ctx, language=req.language)
        else:
            corr = get_corrector(req.method, llm=llm)
            result = corr.correct(ctx, language=req.language)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    metrics = evaluator.evaluate_one(result)
    inject_summary = (
        f"### 注入摘要\n"
        f"- 文档总数：**{len(ctx.docs)}**（positive **{ctx.meta.get('positives',0)}** + noise **{ctx.meta.get('noises',0)}**）\n"
        f"- 实际比例：**{ctx.noise_ratio}** · 类型：**{ctx.noise_type}** · 位置：**{ctx.noise_position}**"
    )
    sys_msg, user_msg = build_naive_prompt(ctx.query, ctx.docs, language=req.language)
    prompt_md = f"**[System]**\n```\n{sys_msg}\n```\n\n**[User]**\n```\n{user_msg}\n```"

    return RunResponse(
        query=record.query,
        gold=" / ".join(record.answers_norm) or "(无)",
        prediction=result.prediction,
        metrics=MetricsOut(
            em=metrics.em,
            contains=metrics.contains,
            token_f1=metrics.token_f1,
            rouge_l=metrics.rouge_l,
            isr=metrics.isr,
            nar=metrics.nar,
            verdict=verdict(metrics),
        ),
        inject_summary=inject_summary,
        injected_html=render_injected_html(ctx),
        prompt_markdown=prompt_md,
        meta={
            "method": req.method,
            "prompt_tokens": result.metadata.get("prompt_tokens", 0),
            "completion_tokens": result.metadata.get("completion_tokens", 0),
            "latency": result.metadata.get("latency", 0.0),
            "cached": result.metadata.get("cached", False),
        },
    )
