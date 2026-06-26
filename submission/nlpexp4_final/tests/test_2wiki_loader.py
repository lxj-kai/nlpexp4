"""Tests for 2WikiMultihopQA loader and noise injection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_loader import load_dataset
from src.noise_injector import inject


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "2wiki"


@pytest.fixture(scope="module")
def require_2wiki() -> Path:
    main = DATA_DIR / "en.json"
    if not main.exists():
        pytest.skip("run `python scripts/prepare_2wiki.py` first")
    return DATA_DIR


def test_2wiki_main_loads(require_2wiki: Path) -> None:
    records = load_dataset("en", "main", dataset="2wiki", limit=20, shuffle=False)
    assert len(records) == 20
    rec = records[0]
    assert rec.dataset == "2wiki"
    assert rec.language == "en"
    assert rec.query
    assert rec.positive
    assert rec.negative
    assert rec.answers_norm


def test_2wiki_fact_loads(require_2wiki: Path) -> None:
    records = load_dataset("en", "fact", dataset="2wiki", limit=10, shuffle=False)
    assert len(records) == 10
    rec = records[0]
    assert rec.has_counterfactual
    assert rec.positive_wrong


def test_2wiki_hard_negatives_in_context(require_2wiki: Path) -> None:
    records = load_dataset("en", "main", dataset="2wiki", limit=5, shuffle=False)
    for rec in records:
        pos_join = " ".join(rec.positive).lower()
        for neg in rec.negative:
            assert neg.strip()
            assert neg.lower() not in pos_join or len(neg) > 40


def test_2wiki_semantic_noise(require_2wiki: Path) -> None:
    records = load_dataset("en", "main", dataset="2wiki", limit=3, shuffle=False)
    ctx = inject(records[0], noise_ratio=0.5, noise_type="semantic")
    assert ctx.docs
    assert any(lab == "negative" for lab in ctx.labels)


def test_2wiki_metadata(require_2wiki: Path) -> None:
    meta = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert meta["source"]
    assert meta["main_size"] >= 100
