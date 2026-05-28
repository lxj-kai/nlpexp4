"""Health check & global config endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from src.correctors import list_correctors
from src.noiser_loader import list_noiser_subsets

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def api_config():
    noiser_subs = list_noiser_subsets()
    return {
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noise_positions": ["front", "back", "interleave", "surround"],
        "methods": ["naive", *list_correctors()],
        "subsets": ["main", "refine", "fact", "int", "custom", *noiser_subs],
        "languages": ["zh", "en"],
    }
