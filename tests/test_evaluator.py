"""Unit tests for src/evaluator.py — EM / Contains / Token-F1 / ROUGE-L / aggregate。"""
from __future__ import annotations

import pytest

from src.evaluator import (
    _best_over_golds,
    _contains_match,
    _exact_match,
    _rouge_l,
    _token_f1,
    aggregate,
    normalize_answer,
)


# ── normalize_answer ──


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("Hello World!") == "helloworld"

    def test_chinese_punctuation(self):
        assert normalize_answer("你好，世界！") == "你好世界"

    def test_empty(self):
        assert normalize_answer("") == ""

    def test_whitespace(self):
        assert normalize_answer("  hello  world  ") == "helloworld"

    def test_quotes(self):
        assert normalize_answer('"test"') == "test"


# ── _exact_match ──


class TestExactMatch:
    def test_match(self):
        assert _exact_match("hello", "Hello") == 1.0

    def test_no_match(self):
        assert _exact_match("hello", "world") == 0.0

    def test_punctuation_ignored(self):
        assert _exact_match("hello!", "hello") == 1.0

    def test_empty_pred(self):
        assert _exact_match("", "hello") == 0.0

    def test_empty_gold(self):
        assert _exact_match("hello", "") == 0.0


# ── _contains_match ──


class TestContainsMatch:
    def test_contains(self):
        assert _contains_match("the answer is hello", "hello") == 1.0

    def test_not_contains(self):
        assert _contains_match("world", "hello") == 0.0

    def test_exact_is_contains(self):
        assert _contains_match("hello", "hello") == 1.0

    def test_empty(self):
        assert _contains_match("", "hello") == 0.0
        assert _contains_match("hello", "") == 0.0


# ── _token_f1 ──


class TestTokenF1:
    def test_exact(self):
        assert _token_f1("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        f1 = _token_f1("hello world foo", "hello world bar")
        assert 0 < f1 < 1

    def test_no_overlap(self):
        assert _token_f1("abc", "xyz") == 0.0

    def test_empty(self):
        assert _token_f1("", "hello") == 0.0
        assert _token_f1("hello", "") == 0.0

    def test_chinese(self):
        f1 = _token_f1("司马懿是魏国重臣", "司马懿是三国人物")
        assert f1 > 0


# ── _rouge_l ──


class TestRougeL:
    def test_identical(self):
        assert _rouge_l("hello world", "hello world") == 1.0

    def test_partial(self):
        rl = _rouge_l("hello world foo", "hello world bar")
        assert 0 < rl < 1

    def test_no_common(self):
        assert _rouge_l("abc", "xyz") == 0.0

    def test_empty(self):
        assert _rouge_l("", "hello") == 0.0
        assert _rouge_l("hello", "") == 0.0

    def test_token_subsequence(self):
        rl = _rouge_l("alpha bravo charlie delta", "alpha charlie echo")
        assert rl > 0

    def test_chinese(self):
        rl = _rouge_l("司马懿是魏国重臣", "司马懿是三国人物")
        assert rl > 0


# ── aggregate ──


class TestAggregate:
    @staticmethod
    def _row(method="naive", ratio=0.0, **kw):
        base = {
            "method": method,
            "noise_ratio": ratio,
            "em": 1.0,
            "contains": 1.0,
            "token_f1": 0.8,
            "rouge_l": 0.7,
            "judge_score": None,
            "isr": 0.5,
            "nar": 0.1,
        }
        base.update(kw)
        return base

    def test_single_group(self):
        rows = [self._row(token_f1=0.8), self._row(token_f1=0.6)]
        result = aggregate(rows, group_by=("method", "noise_ratio"))
        assert len(result) == 1
        assert result[0]["n"] == 2
        assert abs(result[0]["token_f1"] - 0.7) < 1e-4

    def test_multiple_groups(self):
        rows = [self._row(method="naive"), self._row(method="prompt")]
        result = aggregate(rows, group_by=("method",))
        assert len(result) == 2

    def test_none_excluded(self):
        rows = [
            self._row(rouge_l=0.7, judge_score=None),
            self._row(rouge_l=None, judge_score=None),
        ]
        result = aggregate(rows, group_by=("method",))
        assert result[0]["rouge_l"] == 0.7
        assert result[0]["judge_score"] is None

    def test_empty_rows(self):
        assert aggregate([], group_by=("method",)) == []


# ── _best_over_golds ──


class TestBestOverGolds:
    def test_single_gold(self):
        assert _best_over_golds(_exact_match, "hello", ["hello"]) == 1.0

    def test_multiple_golds(self):
        assert _best_over_golds(_exact_match, "world", ["hello", "world"]) == 1.0

    def test_no_match(self):
        assert _best_over_golds(_exact_match, "xyz", ["hello", "world"]) == 0.0

    def test_empty_golds(self):
        assert _best_over_golds(_exact_match, "hello", []) == 0.0
