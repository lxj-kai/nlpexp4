"""Health check & global config endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from src.correctors import list_correctors
from src.dataset_registry import config_payload

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health")
def health():
    from src.config import CONFIG

    return {
        "status": "ok",
        "generation_model": CONFIG.model,
        "generation_api_base": CONFIG.api_base,
        "generation_provider": "lmstudio",
        "judge_model": CONFIG.judge_model,
        "judge_api_base": CONFIG.judge_api_base,
        "judge_provider": "deepseek",
    }


@router.get("/config")
def api_config():
    return {
        **config_payload(),
        "methods": ["naive", *list_correctors()],
    }
