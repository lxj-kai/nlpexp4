"""Health check & global config endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from src.correctors import list_correctors
from src.mobilemem import MOBILEMEM_SUBSETS
from src.noiser_loader import list_noiser_subsets

router = APIRouter(prefix="/api", tags=["meta"])

HIDDEN_METHODS = {
    "ablated_full",
    "ablated_no_decompose",
    "ablated_no_evidence",
    "ablated_no_tag",
}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def api_config():
    noiser_subs = list_noiser_subsets()
    methods = [name for name in list_correctors() if name not in HIDDEN_METHODS]
    return {
        "noise_types": ["semantic", "counterfactual", "mixed"],
        "noise_positions": ["front", "back", "interleave", "surround"],
        "methods": ["naive", *methods],
        "subsets": [
            "main",
            "refine",
            "fact",
            "int",
            *MOBILEMEM_SUBSETS,
            *noiser_subs,
        ],
        "languages": ["zh", "en"],
    }
