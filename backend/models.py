"""Pydantic request / response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SampleItem(BaseModel):
    id: int
    label: str


class SamplesResponse(BaseModel):
    items: list[SampleItem]


class InjectRequest(BaseModel):
    language: Literal["zh", "en"] = "zh"
    subset: str = "main"
    sample_id: int
    method: str = "naive"
    noise_ratio: float = Field(0.5, ge=0.0, le=1.0)
    noise_type: Literal["semantic", "counterfactual", "mixed"] = "semantic"
    noise_position: Literal["front", "back", "interleave", "surround"] = "interleave"


class PromptCallOut(BaseModel):
    title: str
    prompt_markdown: str
    output: str = ""


class InjectResponse(BaseModel):
    summary: str
    injected_html: str
    prompt_markdown: str
    prompt_calls: list[PromptCallOut] = Field(default_factory=list)


class RunRequest(InjectRequest):
    method: str = "naive"


class MetricsOut(BaseModel):
    em: float
    contains: float
    token_f1: float
    rouge_l: float
    judge_score: float | None = None
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
    prompt_calls: list[PromptCallOut] = Field(default_factory=list)
    meta: dict[str, Any]
