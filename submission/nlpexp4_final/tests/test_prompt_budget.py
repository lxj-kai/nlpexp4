"""Unit tests for prompt budget helpers."""
from __future__ import annotations

from src.prompt_budget import (
    count_messages_tokens,
    count_tokens,
    fit_messages_to_budget,
    truncate_content_to_tokens,
    truncate_text,
)


class TestTruncateText:
    def test_keeps_short_text(self):
        assert truncate_text("hello", 100) == "hello"

    def test_truncates_long_text(self):
        out = truncate_text("a" * 100, 50)
        assert len(out) <= 50
        assert out.endswith("...[truncated for context limit]")


class TestFitMessagesToBudget:
    def test_noop_when_within_budget(self):
        msgs = [{"role": "user", "content": "hello"}]
        out, truncated = fit_messages_to_budget(msgs, max_prompt_tokens=1000)
        assert truncated is False
        assert out[0]["content"] == "hello"

    def test_truncates_oversized_user_message(self):
        msgs = [{"role": "user", "content": "word " * 5000}]
        out, truncated = fit_messages_to_budget(msgs, max_prompt_tokens=200)
        assert truncated is True
        assert count_messages_tokens(out) <= 200
        assert "truncated for context limit" in out[0]["content"]


class TestTruncateContentToTokens:
    def test_binary_search_fits_budget(self):
        content = "token " * 8000
        out = truncate_content_to_tokens(content, max_tokens=100)
        assert count_tokens(out) <= 100
        assert out.endswith("...[truncated for context limit]")
