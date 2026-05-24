"""Unit tests for src/noise_injector.py — 噪音注入逻辑。"""
from __future__ import annotations

import random

import pytest

from src.data_loader import RGBRecord
from src.noise_injector import NoisyContext, _arrange, batch_inject, inject


def _rec(
    *,
    id: int = 1,
    n_pos: int = 3,
    n_neg: int = 5,
    n_pw: int = 2,
) -> RGBRecord:
    return RGBRecord(
        id=id,
        query="测试问题",
        answer=["测试答案"],
        positive=[f"正面文档{i}" for i in range(n_pos)],
        negative=[f"噪音文档{i}" for i in range(n_neg)],
        positive_wrong=[f"反事实文档{i}" for i in range(n_pw)],
    )


# ── inject ──


class TestInject:
    def test_zero_noise(self):
        ctx = inject(_rec(), 0.0, seed=42)
        assert ctx.noise_ratio == 0.0
        assert all(l == "positive" for l in ctx.labels)

    def test_full_noise(self):
        ctx = inject(_rec(), 1.0, seed=42)
        assert ctx.noise_ratio == 1.0
        assert all(l != "positive" for l in ctx.labels)

    def test_half_noise_has_both(self):
        ctx = inject(_rec(n_pos=5, n_neg=5), 0.5, seed=42)
        assert sum(1 for l in ctx.labels if l == "positive") >= 1
        assert sum(1 for l in ctx.labels if l != "positive") >= 1

    def test_noise_ratio_bounded(self):
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ctx = inject(_rec(), ratio, seed=42)
            assert 0.0 <= ctx.noise_ratio <= 1.0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            inject(_rec(), -0.1)
        with pytest.raises(ValueError):
            inject(_rec(), 1.5)

    def test_docs_labels_same_length(self):
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ctx = inject(_rec(), ratio, seed=42)
            assert len(ctx.docs) == len(ctx.labels)

    def test_max_docs_respected(self):
        ctx = inject(_rec(n_pos=20, n_neg=20), 0.5, max_docs=5, seed=42)
        assert len(ctx.docs) <= 5

    def test_noise_types(self):
        for ntype in ["semantic", "counterfactual", "mixed"]:
            ctx = inject(_rec(), 0.5, noise_type=ntype, seed=42)
            assert ctx.noise_type == ntype

    def test_noise_positions(self):
        for pos in ["front", "back", "interleave", "surround"]:
            ctx = inject(_rec(), 0.5, noise_position=pos, seed=42)
            assert ctx.noise_position == pos

    def test_counterfactual_fallback(self):
        ctx = inject(_rec(n_pw=0), 0.5, noise_type="counterfactual", seed=42)
        assert all(l in ("positive", "negative") for l in ctx.labels)

    def test_deterministic_with_seed(self):
        a = inject(_rec(), 0.5, seed=42)
        b = inject(_rec(), 0.5, seed=42)
        assert a.docs == b.docs
        assert a.labels == b.labels

    def test_no_positive_raises(self):
        with pytest.raises(ValueError, match="positive"):
            inject(_rec(n_pos=0), 0.5)

    def test_meta_fields(self):
        ctx = inject(_rec(), 0.5, seed=42)
        assert "total" in ctx.meta
        assert "positives" in ctx.meta

    def test_noisy_context_properties(self):
        ctx = inject(_rec(), 0.5, seed=42)
        assert all(ctx.labels[i] == "positive" for i in ctx.positive_indices)
        assert all(ctx.labels[i] != "positive" for i in ctx.noise_indices)


# ── batch_inject ──


class TestBatchInject:
    def test_basic(self):
        records = [_rec(id=i) for i in range(5)]
        ctxs = batch_inject(records, noise_ratio=0.5)
        assert len(ctxs) == 5

    def test_skips_bad_records(self):
        records = [_rec(id=0, n_pos=0), _rec(id=1)]
        ctxs = batch_inject(records, noise_ratio=0.5)
        assert len(ctxs) == 1
        assert ctxs[0].sample_id == 1


# ── _arrange ──


class TestArrange:
    def _rng(self):
        return random.Random(42)

    def test_front(self):
        docs, labels = _arrange(
            ["p1", "p2"], ["n1", "n2"],
            ["positive", "positive"], ["negative", "negative"],
            "front", self._rng(),
        )
        assert docs == ["n1", "n2", "p1", "p2"]
        assert labels == ["negative", "negative", "positive", "positive"]

    def test_back(self):
        docs, _ = _arrange(
            ["p1", "p2"], ["n1", "n2"],
            ["positive", "positive"], ["negative", "negative"],
            "back", self._rng(),
        )
        assert docs == ["p1", "p2", "n1", "n2"]

    def test_surround(self):
        docs, _ = _arrange(
            ["p1"], ["n1", "n2", "n3", "n4"],
            ["positive"], ["negative"] * 4,
            "surround", self._rng(),
        )
        pos_idx = docs.index("p1")
        assert 0 < pos_idx < len(docs) - 1

    def test_interleave_preserves_all(self):
        docs, labels = _arrange(
            ["p1", "p2"], ["n1", "n2"],
            ["positive", "positive"], ["negative", "negative"],
            "interleave", self._rng(),
        )
        assert set(docs) == {"p1", "p2", "n1", "n2"}
        assert len(docs) == 4

    def test_surround_few_noise_degrades(self):
        docs, _ = _arrange(
            ["p1", "p2"], ["n1"],
            ["positive", "positive"], ["negative"],
            "surround", self._rng(),
        )
        assert docs[0] == "n1"
