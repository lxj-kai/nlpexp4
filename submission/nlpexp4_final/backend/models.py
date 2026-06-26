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
    dataset: str = "rgb"
    language: str = "zh"
    subset: str = "main"
    sample_id: int
    noise_ratio: float = Field(0.5, ge=0.0, le=1.0)
    noise_type: str = "semantic"
    noise_position: Literal["front", "back", "interleave", "surround"] = "interleave"


class InjectResponse(BaseModel):
    summary: str
    injected_html: str
    prompt_markdown: str


class RunRequest(InjectRequest):
    method: str = "naive"


class MetricsOut(BaseModel):
    judge_score: float | None = None
    judge_correct: float | None = None
    isr: float
    nar: float
    isr_semantic: float | None = None
    nar_semantic: float | None = None
    em: float | None = None
    contains: float | None = None
    token_f1: float | None = None
    rouge_l: float | None = None
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
