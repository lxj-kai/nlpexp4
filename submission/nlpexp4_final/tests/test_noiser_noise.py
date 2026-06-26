"""NoiserBench 7 类噪声注入测试。"""
from __future__ import annotations

import pytest

from src.data_loader import RGBRecord
from src.noise_injector import inject, noise_types_for_dataset
from src.noiser_loader import NOISER_NOISE_TYPES, iter_noiser_records


def _noiser_rec(**overrides) -> RGBRecord:
    base = RGBRecord(
        id=1,
        query="Who won?",
        answer=["Alice"],
        positive=["Alice won the race in 2020."],
        negative=["Bob ran fast."],
        positive_wrong=["Bob won the race in 2020."],
        fakeanswer="Bob",
        language="en",
        subset="hotpotqa",  # type: ignore[arg-type]
        dataset="noiser_bench",
        noiser_noises={
            "semantic": ["Bob ran fast."],
            "counterfactual": ["Bob won the race in 2020."],
            "supportive": ["Alice won the race in 2020. She trained hard."],
            "orthographic": ["Al1ce w0n the race in 2O2O."],
            "datatype": ["{'winner': 'Alice', 'year': 2020}"],
            "illegal_sentence": ["iors ip ious ml illo chr five"],
            "counterfactual_answer": ["Bob"],
        },
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestNoiserNoiseTypes:
    def test_registry_lists_seven_plus_mixed(self):
        types = noise_types_for_dataset("noiser_bench")
        for t in NOISER_NOISE_TYPES:
            assert t in types
        assert "mixed" in types

    def test_strategyqa_excluded_from_demo_subsets(self):
        from src.dataset_registry import (
            NOISER_BENCH_EXCLUDED_SUBSETS,
            noiser_bench_subsets_available,
        )

        subs = noiser_bench_subsets_available()
        assert "strategyqa" not in subs
        assert "hotpotqa" in subs
        assert "strategyqa" in NOISER_BENCH_EXCLUDED_SUBSETS

    @pytest.mark.parametrize("ntype", NOISER_NOISE_TYPES)
    def test_each_type_injects(self, ntype: str):
        ctx = inject(_noiser_rec(), 0.5, noise_type=ntype, seed=42)
        assert ctx.noise_type == ntype
        assert any(l != "positive" for l in ctx.labels)

    def test_mixed_uses_harmful_only(self):
        ctx = inject(_noiser_rec(), 1.0, noise_type="mixed", seed=42, max_docs=20)
        assert "supportive" not in ctx.labels

    def test_supportive_label(self):
        ctx = inject(_noiser_rec(), 1.0, noise_type="supportive", seed=42, max_docs=5)
        assert all(l == "supportive" for l in ctx.labels)

    def test_real_record_has_all_pools(self):
        rec = next(iter_noiser_records("hotpotqa"))
        assert rec.noiser_noises
        for key in NOISER_NOISE_TYPES:
            assert key in rec.noiser_noises
            assert rec.noiser_noises[key]

    def test_real_record_inject_each_type(self):
        rec = next(iter_noiser_records("hotpotqa"))
        for ntype in NOISER_NOISE_TYPES:
            ctx = inject(rec, 0.5, noise_type=ntype, seed=1, max_docs=8)
            assert len(ctx.docs) >= 2
