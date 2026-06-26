"""Noise-injection & full-pipeline run endpoints."""
from __future__ import annotations

from openai import AuthenticationError, OpenAIError
from fastapi import APIRouter, HTTPException

from backend.deps import evaluator, find_record, llm
from backend.models import (
    InjectRequest,
    InjectResponse,
    MetricsOut,
    PromptCallOut,
    RunRequest,
    RunResponse,
)
from backend.services import render_injected_html, verdict
from src.config import CONFIG
from src.correctors import get_corrector
from src.correctors.adaptive_corrector import (
    NOISE_DETECT_SYSTEM_EN,
    NOISE_DETECT_SYSTEM_ZH,
    NOISE_DETECT_USER_EN,
    NOISE_DETECT_USER_ZH,
)
from src.mobilemem import (
    is_mobilemem_graph_subset,
    is_mobilemem_high_noise_subset,
    is_mobilemem_subset,
)
from src.noise_injector import inject
from src.prompts import (
    COT_EVIDENCE_SYSTEM_EN,
    COT_EVIDENCE_SYSTEM_ZH,
    ITER_FILTER_SYSTEM_EN,
    ITER_FILTER_SYSTEM_ZH,
    ITER_FILTER_USER_TMPL,
    ITER_FILTER_USER_TMPL_EN,
    NAIVE_USER_TMPL,
    PROMPT_AWARE_SYSTEM_EN,
    PROMPT_AWARE_SYSTEM_ZH,
    SELFRAG_REL_SYSTEM_EN,
    SELFRAG_REL_SYSTEM_ZH,
    SELFRAG_REL_USER_TMPL,
    SELFRAG_REL_USER_TMPL_EN,
    VOTE_PROMPTS_ZH,
    build_naive_prompt,
    format_context,
)
from src.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api", tags=["experiment"])


def _messages_prompt_markdown(messages: list[dict]) -> str:
    parts = []
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    for msg in messages:
        role = role_names.get(str(msg.get("role", "")).lower(), str(msg.get("role", "Message")))
        content = msg.get("content", "")
        parts.append(f"**[{role}]**\n```\n{content}\n```")
    return "\n\n".join(parts)


def _guess_call_title(index: int, messages: list[dict]) -> str:
    system = ""
    user = ""
    for msg in messages:
        if msg.get("role") == "system":
            system = str(msg.get("content", ""))
        elif msg.get("role") == "user":
            user = str(msg.get("content", ""))

    joined = f"{system}\n{user}"
    if "文档质量检测器" in joined or "document quality inspector" in joined:
        label = "噪音类型检测"
    elif "文档相关性评估器" in joined or "document relevance assessor" in joined:
        label = "检索质量打分"
    elif "判断给定文档对回答给定问题是否相关" in joined or "Determine if the given document is relevant" in joined:
        label = "相关性门控"
    elif "判断给定答案是否得到给定文档的支撑" in joined or "Determine if the given answer is supported" in joined:
        label = "答案支持性检查"
    elif "答案审阅员" in joined:
        label = "答案一致性自检"
    elif "答案修订员" in joined:
        label = "重读文档修订"
    elif "答案聚合器" in joined:
        label = "候选答案聚合"
    elif "事实核查员" in joined or "多疑的研究员" in joined or "证据链推理者" in joined:
        label = "候选答案生成"
    elif "证据链推理助手" in joined or "evidence-chain reasoning assistant" in joined:
        label = "证据链生成"
    elif "候选答案" in joined:
        label = "候选答案生成"
    else:
        label = "答案生成"
    return f"Call {index} · {label}"


class _TracingLLM:
    def __init__(self, base) -> None:
        self._base = base
        self.calls: list[PromptCallOut] = []

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def chat(self, messages: list[dict], **kwargs) -> dict:
        title = _guess_call_title(len(self.calls) + 1, messages)
        prompt_markdown = _messages_prompt_markdown(messages)
        out = self._base.chat(messages, **kwargs)
        self.calls.append(
            PromptCallOut(
                title=title,
                prompt_markdown=prompt_markdown,
                output=(out.get("content", "") or "").strip(),
            )
        )
        return out


def _inject_kwargs(req: InjectRequest, record) -> dict:
    record_meta = getattr(record, "meta", None) or {}
    kwargs = {
        "noise_ratio": req.noise_ratio,
        "noise_type": req.noise_type,
        "noise_position": req.noise_position,
        "max_docs": CONFIG.max_docs,
    }
    if is_mobilemem_high_noise_subset(req.subset):
        kwargs["max_docs"] = 40
        kwargs["min_positive"] = min(2, len(record.positive))
    if is_mobilemem_graph_subset(req.subset):
        kwargs["max_docs"] = 80
        kwargs["min_positive"] = min(4, len(record.positive))
        kwargs["keep_all_positive"] = True
    return kwargs


def _format_prompt_markdown(system: str, user: str) -> str:
    return f"**[System]**\n```\n{system}\n```\n\n**[User]**\n```\n{user}\n```"


def _method_prompt_markdown(method: str, query: str, docs: list[str], language: str) -> str:
    if method == "prompt":
        system = PROMPT_AWARE_SYSTEM_ZH if language == "zh" else PROMPT_AWARE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
        return _format_prompt_markdown(system, user)

    if method == "confidence":
        system = COT_EVIDENCE_SYSTEM_ZH if language == "zh" else COT_EVIDENCE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
        return _format_prompt_markdown(system, user)

    system, user = build_naive_prompt(query, docs, language=language)
    return _format_prompt_markdown(system, user)


def _first_call_messages(method: str, query: str, docs: list[str], language: str) -> list[dict]:
    if method == "prompt":
        system = PROMPT_AWARE_SYSTEM_ZH if language == "zh" else PROMPT_AWARE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
    elif method == "confidence":
        system = COT_EVIDENCE_SYSTEM_ZH if language == "zh" else COT_EVIDENCE_SYSTEM_EN
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
    elif method == "iterative":
        system = ITER_FILTER_SYSTEM_ZH if language == "zh" else ITER_FILTER_SYSTEM_EN
        tmpl = ITER_FILTER_USER_TMPL if language == "zh" else ITER_FILTER_USER_TMPL_EN
        first_doc = docs[0] if docs else ""
        user = tmpl.format(query=query, doc=first_doc[:1500])
    elif method == "selfrag":
        system = SELFRAG_REL_SYSTEM_ZH if language == "zh" else SELFRAG_REL_SYSTEM_EN
        tmpl = SELFRAG_REL_USER_TMPL if language == "zh" else SELFRAG_REL_USER_TMPL_EN
        first_doc = docs[0] if docs else ""
        user = tmpl.format(query=query, doc=first_doc[:1500])
    elif method == "voting":
        system = VOTE_PROMPTS_ZH[0]
        user = NAIVE_USER_TMPL.format(query=query, n=len(docs), context=format_context(docs))
    elif method == "adaptive":
        is_zh = language == "zh"
        system = NOISE_DETECT_SYSTEM_ZH if is_zh else NOISE_DETECT_SYSTEM_EN
        tmpl = NOISE_DETECT_USER_ZH if is_zh else NOISE_DETECT_USER_EN
        user = tmpl.format(query=query, context=format_context(docs, max_chars_per_doc=1200))
    elif method == "iterative_sc":
        system = "你是严谨的问答助手。请仅基于提供的文档作答，答案尽量简短。"
        user = f"【问题】{query}\n\n【文档】共{len(docs)}篇：\n{format_context(docs)}\n\n请简短作答："
    else:
        system, user = build_naive_prompt(query, docs, language=language)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _preview_calls(method: str, query: str, docs: list[str], language: str) -> list[PromptCallOut]:
    messages = _first_call_messages(method, query, docs, language)
    return [
        PromptCallOut(
            title=_guess_call_title(1, messages),
            prompt_markdown=_messages_prompt_markdown(messages),
        )
    ]


@router.post("/inject", response_model=InjectResponse)
def api_inject(req: InjectRequest):
    record = find_record(req.language, req.subset, req.sample_id)
    if is_mobilemem_subset(req.subset) and req.noise_type != "semantic":
        raise HTTPException(
            400,
            detail="MobileMem synthetic memories only support semantic non-contradictory noise.",
        )
    try:
        ctx = inject(record, **_inject_kwargs(req, record))
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
    prompt_calls = _preview_calls(req.method, ctx.query, ctx.docs, req.language)
    prompt_md = prompt_calls[0].prompt_markdown if prompt_calls else _method_prompt_markdown(
        req.method, ctx.query, ctx.docs, req.language
    )
    return InjectResponse(
        summary=summary,
        injected_html=render_injected_html(ctx),
        prompt_markdown=prompt_md,
        prompt_calls=prompt_calls,
    )


@router.post("/run", response_model=RunResponse)
def api_run(req: RunRequest):
    record = find_record(req.language, req.subset, req.sample_id)
    if is_mobilemem_subset(req.subset) and req.noise_type != "semantic":
        raise HTTPException(
            400,
            detail="MobileMem synthetic memories only support semantic non-contradictory noise.",
        )
    try:
        ctx = inject(record, **_inject_kwargs(req, record))
        trace_llm = _TracingLLM(llm)
        if req.method == "naive":
            rag = RAGPipeline(llm=trace_llm)
            result = rag.answer(ctx, language=req.language)
        else:
            corr = get_corrector(req.method, llm=trace_llm)
            result = corr.correct(ctx, language=req.language)
    except AuthenticationError:
        raise HTTPException(
            401,
            detail=(
                "LLM authentication failed. Check that DEEPSEEK_API_KEY is set "
                "in the backend process environment."
            ),
        )
    except OpenAIError as e:
        raise HTTPException(502, detail=f"LLM API request failed: {e}")
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    metrics = evaluator.evaluate_one(result)
    inject_summary = (
        f"### 注入摘要\n"
        f"- 文档总数：**{len(ctx.docs)}**（positive **{ctx.meta.get('positives',0)}** + noise **{ctx.meta.get('noises',0)}**）\n"
        f"- 实际比例：**{ctx.noise_ratio}** · 类型：**{ctx.noise_type}** · 位置：**{ctx.noise_position}**"
    )
    prompt_calls = trace_llm.calls
    prompt_md = prompt_calls[0].prompt_markdown if prompt_calls else _method_prompt_markdown(
        req.method, ctx.query, ctx.docs, req.language
    )

    return RunResponse(
        query=record.query,
        gold=" / ".join(record.answers_norm) or "(无)",
        prediction=result.prediction,
        metrics=MetricsOut(
            em=metrics.em,
            contains=metrics.contains,
            token_f1=metrics.token_f1,
            rouge_l=metrics.rouge_l,
            judge_score=metrics.judge_score,
            isr=metrics.isr,
            nar=metrics.nar,
            verdict=verdict(metrics),
        ),
        inject_summary=inject_summary,
        injected_html=render_injected_html(ctx),
        prompt_markdown=prompt_md,
        prompt_calls=prompt_calls,
        meta={
            "method": req.method,
            "prompt_tokens": result.metadata.get("prompt_tokens", 0),
            "completion_tokens": result.metadata.get("completion_tokens", 0),
            "latency": result.metadata.get("latency", 0.0),
            "cached": result.metadata.get("cached", False),
        },
    )
