"""Unit tests for src/metrics.py — ISR/NAR 归因、NS/NRS/CRR 指标。"""
from __future__ import annotations

import pytest

from src.metrics import (
    _is_cjk,
    _safe_div,
    attribute_answer,
    attribution_grams,
    attribution_grams_set,
    correction_recovery_rate,
    noise_resistance_slope,
    noise_sensitivity,
    token_set,
    tokenize,
)


# ── tokenize ──


class TestTokenize:
    def test_english(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_chinese(self):
        assert tokenize("你好世界") == ["你", "好", "世", "界"]

    def test_mixed(self):
        tokens = tokenize("Hello你好World")
        assert "hello" in tokens
        assert "world" in tokens
        assert "你" in tokens

    def test_empty(self):
        assert tokenize("") == []

    def test_numbers(self):
        tokens = tokenize("abc123 xyz")
        assert "abc123" in tokens
        assert "xyz" in tokens

    def test_punctuation_stripped(self):
        tokens = tokenize("hello, world!")
        assert tokens == ["hello", "world"]


class TestTokenSet:
    def test_dedup(self):
        s = token_set("hello hello world")
        assert s == {"hello", "world"}


# ── _is_cjk ──


class TestIsCjk:
    def test_chinese_char(self):
        assert _is_cjk("你") is True

    def test_english_word(self):
        assert _is_cjk("hello") is False

    def test_empty(self):
        assert _is_cjk("") is False


# ── attribution_grams ──


class TestAttributionGrams:
    def test_chinese_bigrams(self):
        grams = attribution_grams("司马懿")
        assert "司马" in grams
        assert "马懿" in grams

    def test_chinese_stopchars_filtered(self):
        grams = attribution_grams("的是在")
        assert len(grams) == 0

    def test_english_stopwords_filtered(self):
        grams = attribution_grams("the quick brown fox")
        assert "the" not in grams
        assert "quick" in grams
        assert "brown" in grams
        assert "fox" in grams

    def test_single_char_english_dropped(self):
        grams = attribution_grams("a b c long")
        assert "a" not in grams
        assert "long" in grams

    def test_empty(self):
        assert attribution_grams("") == []

    def test_mixed_chinese_english(self):
        grams = attribution_grams("Hello司马懿World")
        assert "hello" in grams
        assert "司马" in grams
        assert "马懿" in grams
        assert "world" in grams

    def test_set_dedup(self):
        s = attribution_grams_set("司马懿司马懿")
        assert "司马" in s
        assert len(s) == len(set(s))


# ── attribute_answer (ISR / NAR) ──


class TestAttributeAnswer:
    def test_chinese_basic(self):
        attr = attribute_answer(
            "司马懿",
            ["司马懿（179-251）是三国时期著名军事家，魏国重臣。"],
            ["positive"],
        )
        assert attr.n_answer_grams > 0
        assert attr.isr > 0

    def test_chinese_positive_vs_negative(self):
        attr = attribute_answer(
            "司马懿",
            ["司马懿（179-251）是三国时期著名军事家。", "诸葛亮是蜀国丞相。"],
            ["positive", "negative"],
        )
        assert attr.isr > 0
        assert attr.n_from_positive > 0

    def test_label_swap_flips_isr_nar(self):
        docs = ["司马懿是魏国重臣。", "完全无关的内容，毫无联系。"]
        a = attribute_answer("司马懿", docs, ["positive", "negative"])
        b = attribute_answer("司马懿", docs, ["negative", "positive"])
        assert a.isr >= b.isr

    def test_empty_answer(self):
        attr = attribute_answer("", ["doc1"], ["positive"])
        assert attr.n_answer_grams == 0
        assert attr.isr == 0.0
        assert attr.nar == 0.0

    def test_empty_docs(self):
        attr = attribute_answer("hello world", [], [])
        assert attr.n_answer_grams > 0
        assert attr.n_from_positive == 0
        assert attr.n_from_negative == 0

    def test_n_answer_tokens_compat(self):
        attr = attribute_answer("hello world", ["hello"], ["positive"])
        assert attr.n_answer_tokens == attr.n_answer_grams

    def test_sum_equals_total(self):
        attr = attribute_answer(
            "司马懿是三国人物",
            ["司马懿是魏国重臣。", "诸葛亮是蜀国丞相。"],
            ["positive", "negative"],
        )
        assert attr.n_from_positive + attr.n_from_negative + attr.n_from_neither == attr.n_answer_grams


# ── NS / NRS ──


class TestNoiseSensitivity:
    def test_basic(self):
        assert abs(noise_sensitivity(1.0, 0.5) - 0.5) < 1e-9

    def test_no_drop(self):
        assert abs(noise_sensitivity(0.8, 0.8)) < 1e-9

    def test_zero_clean(self):
        assert noise_sensitivity(0.0, 0.5) == 0.0


class TestNoiseResistanceSlope:
    def test_flat(self):
        assert abs(noise_resistance_slope([0.0, 0.5, 1.0], [0.8, 0.8, 0.8])) < 1e-6

    def test_negative_slope(self):
        assert noise_resistance_slope([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]) < 0

    def test_single_point(self):
        assert noise_resistance_slope([0.5], [0.8]) == 0.0

    def test_two_points(self):
        assert abs(noise_resistance_slope([0.0, 1.0], [1.0, 0.0]) - (-1.0)) < 1e-6


# ── CRR ──


class TestCorrectionRecoveryRate:
    def test_full_recovery(self):
        assert abs(correction_recovery_rate(1.0, 0.5, 1.0) - 1.0) < 1e-9

    def test_no_recovery(self):
        assert abs(correction_recovery_rate(1.0, 0.5, 0.5)) < 1e-9

    def test_over_recovery(self):
        assert correction_recovery_rate(0.8, 0.4, 1.0) > 1.0

    def test_no_degradation(self):
        assert correction_recovery_rate(0.8, 0.8, 0.9) == 0.0


# ── _safe_div ──


class TestSafeDiv:
    def test_normal(self):
        assert _safe_div(10, 2) == 5.0

    def test_zero_denom(self):
        assert _safe_div(10, 0) == 0.0

    def test_custom_default(self):
        assert _safe_div(10, 0, default=-1.0) == -1.0
