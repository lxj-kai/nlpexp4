"""Tests for CmedqaRetrieval dataset integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import load_dataset
from src.noise_injector import batch_inject


@pytest.fixture(scope="module")
def cmedqa_main_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "cmedqa" / "zh.json"
    if not path.exists():
        pytest.skip("Cmedqa data missing; run scripts/prepare_cmedqa.py first")
    return path


def test_cmedqa_main_loads(cmedqa_main_path: Path) -> None:
    records = load_dataset("zh", "main", dataset="cmedqa", limit=20, shuffle=False)
    assert len(records) == 20
    assert all(r.positive for r in records)
    assert all(len(r.negative) >= 4 for r in records)


def test_cmedqa_fact_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "cmedqa" / "zh_fact.json"
    if not path.exists():
        pytest.skip("Cmedqa fact data missing")
    records = load_dataset("zh", "fact", dataset="cmedqa", limit=10, shuffle=False)
    assert len(records) == 10
    assert all(r.positive_wrong for r in records)
    assert all(r.fakeanswer for r in records)


def test_cmedqa_noise_injection(cmedqa_main_path: Path) -> None:
    records = load_dataset("zh", "main", dataset="cmedqa", limit=5, shuffle=False)
    ctxs = batch_inject(records, noise_ratio=0.5, noise_type="semantic", seed=42)
    assert len(ctxs) == 5
    assert any(ctx.labels.count("negative") > 0 for ctx in ctxs)
