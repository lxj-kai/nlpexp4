"""Multihop decompose corrector — for 2Wiki-style multi-hop QA.

Pipeline (2 API calls, keeps full docs in the solve step):
  1) Decompose into 1-2 ordered sub-questions
  2) Solve all hops + final answer in ONE pass with documents present

Designed for small LMs (0.5B): fewer calls, context not dropped between hops.
"""
from __future__ import annotations

import re
import time

from ..noise_injector import NoisyContext
from ..prompts import (
    MULTIHOP_DECOMPOSE_SYSTEM_EN,
    MULTIHOP_DECOMPOSE_USER_EN,
    MULTIHOP_SOLVE_SYSTEM_EN,
    MULTIHOP_SOLVE_USER_EN,
    format_context,
)
from ..rag_pipeline import RAGResult
from .base import BaseCorrector, register_corrector

_ANS_PAT = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
_SUBQ_PAT = re.compile(r"^\s*\d+[\).\:\-]\s*(.+)$")
_QUOTED_PAT = re.compile(r'^["\'](.+)["\']$')
_FINAL_PREFIX_PAT = re.compile(
    r"^(?:the final (?:concise )?answer is|final answer[:：])\s*",
    re.IGNORECASE,
)


def parse_subquestions(text: str, *, max_hops: int = 2) -> list[str]:
    """Parse numbered sub-questions from decompose output."""
    out: list[str] = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SUBQ_PAT.match(line)
        q = m.group(1).strip() if m else line
        if len(q) >= 8:
            out.append(q)
    if not out:
        compact = " ".join((text or "").split())
        if compact:
            out.append(compact[:300])
    return out[:max_hops]


def extract_answer_tag(raw: str) -> str:
    if not raw:
        return ""
    m = _ANS_PAT.search(raw)
    if m:
        ans = m.group(1).strip()
    else:
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        ans = lines[-1] if lines else raw.strip()
    ans = _FINAL_PREFIX_PAT.sub("", ans).strip()
    m2 = _QUOTED_PAT.match(ans)
    if m2:
        ans = m2.group(1).strip()
    return ans.strip().strip(".").strip('"').strip("'")


@register_corrector("multihop")
class MultihopDecomposeCorrector(BaseCorrector):
    """Multi-hop: decompose → single-shot hop solve with documents."""

    api_cost = 2
    method_alias = "Multihop-Decompose"
    max_hops: int = 2

    def _chat(self, system: str, user: str, *, max_tokens: int) -> dict:
        return self.llm.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )

    def correct(self, ctx: NoisyContext, *, language: str = "zh") -> RAGResult:
        t0 = time.perf_counter()
        lang = "en" if language == "en" else language
        docs_ctx = format_context(ctx.docs, language=lang)
        api_calls = 0
        prompt_tokens = 0
        completion_tokens = 0

        dec = self._chat(
            MULTIHOP_DECOMPOSE_SYSTEM_EN,
            MULTIHOP_DECOMPOSE_USER_EN.format(query=ctx.query),
            max_tokens=128,
        )
        api_calls += 1
        prompt_tokens += dec.get("prompt_tokens", 0)
        completion_tokens += dec.get("completion_tokens", 0)
        subqs = parse_subquestions(dec.get("content") or "", max_hops=self.max_hops)
        if not subqs:
            subqs = [ctx.query]

        subq_block = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(subqs))
        solve = self._chat(
            MULTIHOP_SOLVE_SYSTEM_EN,
            MULTIHOP_SOLVE_USER_EN.format(
                context=docs_ctx,
                query=ctx.query,
                subquestions=subq_block,
            ),
            max_tokens=384,
        )
        api_calls += 1
        prompt_tokens += solve.get("prompt_tokens", 0)
        completion_tokens += solve.get("completion_tokens", 0)
        solve_raw = solve.get("content") or ""
        prediction = extract_answer_tag(solve_raw)

        return RAGResult(
            sample_id=ctx.sample_id,
            query=ctx.query,
            gold_answers=ctx.gold_answers,
            prediction=prediction,
            docs=ctx.docs,
            labels=list(ctx.labels),
            noise_ratio=ctx.noise_ratio,
            noise_type=ctx.noise_type,
            noise_position=ctx.noise_position,
            metadata={
                "method": self.name,
                "method_alias": self.method_alias,
                "subquestions": subqs,
                "decompose_raw": dec.get("content", ""),
                "solve_raw": solve_raw,
                "api_calls": api_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": time.perf_counter() - t0,
                **ctx.meta,
            },
        )
