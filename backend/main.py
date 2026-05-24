"""FastAPI entrypoint — app creation, CORS middleware, router registration."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routes.experiment import router as experiment_router
from backend.routes.health import router as health_router
from backend.routes.samples import router as samples_router


def _allowed_origins() -> list[str]:
    raw = os.getenv("NLP4_ALLOW_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    return [s.strip() for s in raw.split(",") if s.strip()]


app = FastAPI(title="nlpexp4 API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health_router)
app.include_router(samples_router)
app.include_router(experiment_router)
