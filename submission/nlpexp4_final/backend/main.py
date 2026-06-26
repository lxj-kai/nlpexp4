"""FastAPI backend for nlpexp4 Vue frontend."""
from __future__ import annotations

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

app = FastAPI(title="nlpexp4 API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(samples_router)
app.include_router(experiment_router)
