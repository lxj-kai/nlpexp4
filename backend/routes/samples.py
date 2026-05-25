"""Sample listing & detail endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend.deps import find_record, get_records
from backend.models import SampleItem, SamplesResponse
from backend.services import render_retrieval_html

router = APIRouter(prefix="/api", tags=["samples"])


@router.get("/samples", response_model=SamplesResponse)
def api_samples(language: str = "zh", subset: str = "main"):
    recs = get_records(language, subset)
    items = [SampleItem(id=r.id, label=f"#{r.id} | {r.query[:48]}") for r in recs]
    return SamplesResponse(items=items)


@router.get("/sample/{sample_id}")
def api_sample(sample_id: int, language: str = "zh", subset: str = "main"):
    record = find_record(language, subset, sample_id)
    return {
        "id": record.id,
        "query": record.query,
        "gold": " / ".join(record.answers_norm) or "(无)",
        "retrieval_html": render_retrieval_html(record),
        "counts": {
            "positive": len(record.positive),
            "negative": len(record.negative),
            "positive_wrong": len(record.positive_wrong),
        },
    }
