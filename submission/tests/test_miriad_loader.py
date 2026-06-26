"""Tests for full MIRIAD integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data_loader import load_dataset
from src.miriad_store import EXPECTED_SHARDS, verify_installation
from src.noise_injector import batch_inject


def _require_full_miriad() -> Path:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "miriad" / "raw"
    try:
        verify_installation(raw_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"full MIRIAD not installed: {exc}")
    return raw_dir


def test_miriad_full_installation() -> None:
    raw_dir = _require_full_miriad()
    report = verify_installation(raw_dir)
    assert report["shards"] == EXPECTED_SHARDS
    assert report["bytes"] >= 7_000_000_000


def test_miriad_load_requires_limit() -> None:
    with pytest.raises(ValueError, match="requires explicit limit"):
        load_dataset("en", "main", dataset="miriad")


def test_miriad_main_loads() -> None:
    _require_full_miriad()
    records = load_dataset("en", "main", dataset="miriad", limit=20, shuffle=False)
    assert len(records) == 20
    assert all(r.positive for r in records)
    assert all(len(r.negative) >= 4 for r in records)


def test_miriad_fact_loads() -> None:
    _require_full_miriad()
    records = load_dataset("en", "fact", dataset="miriad", limit=10, shuffle=False)
    assert len(records) == 10
    assert all(r.positive_wrong for r in records)
    assert all(r.fakeanswer for r in records)


def test_miriad_noise_injection() -> None:
    _require_full_miriad()
    records = load_dataset("en", "main", dataset="miriad", limit=5, shuffle=False)
    ctxs = batch_inject(records, noise_ratio=0.5, noise_type="semantic", seed=42)
    assert len(ctxs) == 5
    assert all(ctx.labels.count("negative") > 0 for ctx in ctxs)
