"""Health check & global config endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from src.correctors import list_correctors

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def api_config():
    return {
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noise_positions": ["front", "back", "interleave", "surround"],
        "methods": ["naive", *list_correctors()],
        "subsets": ["main", "refine", "fact", "int"],
        "languages": ["zh", "en"],
    }
