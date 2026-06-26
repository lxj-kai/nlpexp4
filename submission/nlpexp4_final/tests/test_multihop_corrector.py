"""Tests for multihop corrector helpers."""
from __future__ import annotations

from src.correctors.multihop_corrector import extract_answer_tag, parse_subquestions


class TestParseSubquestions:
    def test_numbered(self):
        text = "1. Who founded Les Films Du Losange?\n2. Where does that person work?"
        qs = parse_subquestions(text)
        assert len(qs) == 2
        assert "founded" in qs[0]

    def test_max_hops(self):
        text = "1. Question alpha here\n2. Question beta here\n3. Question gamma here"
        assert len(parse_subquestions(text, max_hops=2)) == 2


class TestExtractAnswerTag:
    def test_tag(self):
        assert extract_answer_tag("Final: <answer>Cahiers du cinéma</answer>") == "Cahiers du cinéma"

    def test_last_line(self):
        assert extract_answer_tag("reason\nyes") == "yes"

    def test_final_prefix(self):
        assert extract_answer_tag('The final concise answer is "Folly To Be Wise".') == "Folly To Be Wise"
