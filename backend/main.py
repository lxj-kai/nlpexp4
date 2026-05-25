"""FastAPI backend for nlpexp4 Vue frontend."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.correctors import get_corrector, list_correctors
from src.data_loader import load_dataset, RGBRecord
from src.evaluator import Evaluator
from src.llm_client import LLMClient
from src.noise_injector import inject
from src.prompts import build_naive_prompt
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="nlpexp4 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm = LLMClient()
_eval = Evaluator(use_llm_judge=False)
_records_cache: dict[tuple[str, str], list] = {}


# ─── Pydantic models ───

class SampleItem(BaseModel):
    id: int
    label: str


class SamplesResponse(BaseModel):
    items: list[SampleItem]


class InjectRequest(BaseModel):
    language: Literal["zh", "en"] = "zh"
    subset: Literal["main", "refine", "fact", "int"] = "main"
    sample_id: int
    noise_ratio: float = Field(0.5, ge=0.0, le=1.0)
    noise_type: Literal["semantic", "counterfactual", "mixed"] = "semantic"
    noise_position: Literal["front", "back", "interleave", "surround"] = "interleave"


class InjectResponse(BaseModel):
    summary: str
    injected_html: str
    prompt_markdown: str


class RunRequest(InjectRequest):
    method: str = "naive"


class MetricsOut(BaseModel):
    em: float
    contains: float
    token_f1: float
    rouge_l: float
    isr: float
    nar: float
    verdict: str


class RunResponse(BaseModel):
    query: str
    gold: str
    prediction: str
    metrics: MetricsOut
    inject_summary: str
    injected_html: str
    prompt_markdown: str
    meta: dict[str, Any]


# ─── Helpers ───

def _get_records(language: str, subset: str) -> list:
    key = (language, subset)
    if key not in _records_cache:
        _records_cache[key] = load_dataset(language=language, subset=subset)
    return _records_cache[key]


def _find_record(language: str, subset: str, sample_id: int) -> RGBRecord:
    for r in _get_records(language, subset):
        if r.id == sample_id:
            return r
    raise HTTPException(404, detail=f"sample {sample_id} not found")


def _render_retrieval_html(record: RGBRecord) -> str:
    parts = [
        "<div class='stage-summary'>"
        f"原始检索池 · positive=<b>{len(record.positive)}</b> · "
        f"negative=<b>{len(record.negative)}</b> · "
        f"positive_wrong=<b>{len(record.positive_wrong)}</b></div>"
    ]
    for i, d in enumerate(record.positive):
        parts.append(_doc_card_html(i, d, "positive"))
    for i, d in enumerate(record.negative):
        parts.append(_doc_card_html(i, d, "negative"))
    for i, d in enumerate(record.positive_wrong):
        parts.append(_doc_card_html(i, d, "positive_wrong"))
    return "".join(parts)


def _doc_card_html(idx: int, doc: str, label: str, max_chars: int = 500) -> str:
    import html as html_mod

    text = doc[:max_chars] + ("…" if len(doc) > max_chars else "")
    safe = html_mod.escape(text).replace("\n", "<br>")
    if label == "positive":
        border, lc = "#15803d", "#166534"
    elif label == "positive_wrong":
        border, lc = "#b45309", "#92400e"
    else:
        border, lc = "#b91c1c", "#991b1b"
    badge = {"positive": "POSITIVE", "negative": "NEGATIVE", "positive_wrong": "COUNTERFACTUAL"}.get(
        label, label.upper()
    )
    return (
        f"<article class='doc-card' style='border-left-color:{border};'>"
        f"<header class='doc-meta' style='color:{lc};'>文档 [{idx}] · {badge}</header>"
        f"<div class='doc-body'>{safe}</div>"
        f"</article>"
    )


def _render_injected_html(ctx) -> str:
    pos = ctx.meta.get("positives", 0)
    noise = ctx.meta.get("noises", len(ctx.docs) - pos)
    parts = [
        "<div class='stage-summary'>"
        f"注入后送给 LLM 的 <b>{len(ctx.docs)}</b> 篇 · "
        f"positive=<b>{pos}</b> / noise=<b>{noise}</b> · 实际比例 = "
        f"<b>{ctx.noise_ratio}</b> · 位置策略 = <b>{ctx.noise_position}</b></div>"
    ]
    for i, (d, lab) in enumerate(zip(ctx.docs, ctx.labels)):
        parts.append(_doc_card_html(i, d, lab))
    return "".join(parts)


def _verdict(metrics) -> str:
    if metrics.contains >= 0.9:
        return "correct"
    if metrics.token_f1 >= 0.5:
        return "partial"
    if metrics.nar > metrics.isr and metrics.nar > 0.3:
        return "noise_biased"
    return "wrong"


# ─── Routes ───

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def api_config():
    return {
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noise_positions": ["front", "back", "interleave", "surround"],
        "methods": ["naive", *list_correctors()],
        "subsets": ["main", "refine", "fact", "int"],
        "languages": ["zh", "en"],
    }


@app.get("/api/samples", response_model=SamplesResponse)
def api_samples(language: str = "zh", subset: str = "main"):
    recs = _get_records(language, subset)
    items = [SampleItem(id=r.id, label=f"#{r.id} | {r.query[:48]}") for r in recs]
    return SamplesResponse(items=items)


@app.get("/api/sample/{sample_id}")
def api_sample(sample_id: int, language: str = "zh", subset: str = "main"):
    record = _find_record(language, subset, sample_id)
    return {
        "id": record.id,
        "query": record.query,
        "gold": " / ".join(record.answers_norm) or "(无)",
        "retrieval_html": _render_retrieval_html(record),
        "counts": {
            "positive": len(record.positive),
            "negative": len(record.negative),
            "positive_wrong": len(record.positive_wrong),
        },
    }


@app.post("/api/inject", response_model=InjectResponse)
def api_inject(req: InjectRequest):
    record = _find_record(req.language, req.subset, req.sample_id)
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
        injected_html=_render_injected_html(ctx),
        prompt_markdown=prompt_md,
    )


@app.post("/api/run", response_model=RunResponse)
def api_run(req: RunRequest):
    record = _find_record(req.language, req.subset, req.sample_id)
    try:
        ctx = inject(
            record,
            noise_ratio=req.noise_ratio,
            noise_type=req.noise_type,
            noise_position=req.noise_position,
            max_docs=CONFIG.max_docs,
        )
        if req.method == "naive":
            rag = RAGPipeline(llm=_llm)
            result = rag.answer(ctx, language=req.language)
        else:
            corr = get_corrector(req.method, llm=_llm)
            result = corr.correct(ctx, language=req.language)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    metrics = _eval.evaluate_one(result)
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
            verdict=_verdict(metrics),
        ),
        inject_summary=inject_summary,
        injected_html=_render_injected_html(ctx),
        prompt_markdown=prompt_md,
        meta={
            "method": req.method,
            "prompt_tokens": result.metadata.get("prompt_tokens", 0),
            "completion_tokens": result.metadata.get("completion_tokens", 0),
            "latency": result.metadata.get("latency", 0.0),
            "cached": result.metadata.get("cached", False),
        },
    )
