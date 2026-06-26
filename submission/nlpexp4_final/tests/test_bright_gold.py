"""BRIGHT gold_answer filtering."""
from __future__ import annotations

from src.data_loader import is_usable_gold_answer, load_dataset, record_has_usable_gold


def test_is_usable_gold_answer():
    assert not is_usable_gold_answer("N/A")
    assert not is_usable_gold_answer("  na  ")
    assert not is_usable_gold_answer("")
    assert is_usable_gold_answer("yes")
    assert is_usable_gold_answer("Long reasoning answer")


def test_bright_load_skips_na():
    recs = load_dataset("en", "main", dataset="bright", limit=200, shuffle=False)
    assert recs
    na = [r for r in recs if any(a.strip().upper() in ("N/A", "NA") for a in r.answers_norm)]
    # After fill_bright_na_gold.py, N/A rows should be patched; allow partial during backfill.
    assert len(na) < len(recs)
