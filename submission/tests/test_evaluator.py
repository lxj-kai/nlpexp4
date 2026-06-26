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
    cap_judge_score_for_abbreviated_pred,
    judge_correct_from_score,
    normalize_answer,
    parse_judge_score,
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
            "judge_score": 0.8,
            "judge_correct": 1.0,
            "em": None,
            "contains": None,
            "token_f1": None,
            "rouge_l": None,
            "isr": 0.5,
            "nar": 0.1,
        }
        base.update(kw)
        return base

    def test_single_group(self):
        rows = [self._row(judge_score=0.8), self._row(judge_score=0.6)]
        result = aggregate(rows, group_by=("method", "noise_ratio"))
        assert len(result) == 1
        assert result[0]["n"] == 2
        assert abs(result[0]["judge_score"] - 0.7) < 1e-4

    def test_multiple_groups(self):
        rows = [self._row(method="naive"), self._row(method="prompt")]
        result = aggregate(rows, group_by=("method",))
        assert len(result) == 2

    def test_none_excluded(self):
        rows = [
            self._row(judge_score=0.7, em=None),
            self._row(judge_score=None, em=None),
        ]
        result = aggregate(rows, group_by=("method",))
        assert result[0]["judge_score"] == 0.7
        assert result[0]["em"] is None

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


class TestParseJudgeScore:
    def test_integer(self):
        assert parse_judge_score("5") == 1.0
        assert parse_judge_score("score: 3") == 0.6

    def test_correct_keyword(self):
        assert parse_judge_score("CORRECT") == 1.0
        assert parse_judge_score("INCORRECT") == 0.0
        assert parse_judge_score("The answer is CORRECT.") is None

    def test_empty(self):
        assert parse_judge_score("") is None

    def test_judge_correct_threshold(self):
        assert judge_correct_from_score(0.8) == 1.0
        assert judge_correct_from_score(0.6) == 0.0


class TestCapAbbreviatedJudgeScore:
    GOLD = (
        "这个现在的孕囊大小与停经56天还是比较相符的，但是没有胚芽和原始心管搏动，"
        "那么就不正常了，多数可能是胚胎停止发育了。"
    )

    def test_caps_conclusion_only_answer(self):
        score = cap_judge_score_for_abbreviated_pred("胚胎停止发育", [self.GOLD], 1.0)
        assert score == 0.6
        assert judge_correct_from_score(score) == 0.0

    def test_keeps_full_answer(self):
        score = cap_judge_score_for_abbreviated_pred(self.GOLD, [self.GOLD], 1.0)
        assert score == 1.0

    def test_keeps_short_gold(self):
        score = cap_judge_score_for_abbreviated_pred("北京", ["北京"], 1.0)
        assert score == 1.0

    def test_skips_cap_for_long_form_dataset(self):
        gold = "A long reasoning paragraph explaining mechanisms in detail " * 3
        score = cap_judge_score_for_abbreviated_pred("short tag", [gold], 1.0, dataset="bright")
        assert score == 1.0


class TestEmptyPrediction:
    def test_skips_judge_and_marks_wrong(self):
        from src.evaluator import Evaluator
        from src.rag_pipeline import RAGResult

        ev = Evaluator(use_semantic_attribution=False)
        calls: list[int] = []

        def _fake_judge(*_args, **_kwargs):
            calls.append(1)
            return 1.0

        ev._llm_judge = _fake_judge  # type: ignore[method-assign]
        result = RAGResult(
            sample_id=0,
            query="test?",
            gold_answers=["answer"],
            prediction="",
            docs=["doc"],
            labels=["positive"],
            noise_ratio=0.0,
            noise_type="semantic",
            noise_position="interleave",
            metadata={"method": "naive"},
        )
        metrics = ev.evaluate_one(result, language="zh")
        assert calls == []
        assert metrics.judge_score == 0.0
        assert metrics.judge_correct == 0.0
