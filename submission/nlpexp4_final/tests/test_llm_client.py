"""Unit tests for Gemma 4 response extraction in src/llm_client.py."""
from __future__ import annotations

from types import SimpleNamespace

from src.llm_client import extract_gemma4_content, is_gemma4_model


class TestIsGemma4Model:
    def test_detects_common_ids(self):
        assert is_gemma4_model("google/gemma-4-e4b")
        assert is_gemma4_model("gemma4:12b")
        assert not is_gemma4_model("qwen2.5-0.5b-instruct-mlx")
        assert not is_gemma4_model("google/gemma-2-9b")


class TestExtractGemma4Content:
    def test_prefers_content_field(self):
        msg = SimpleNamespace(content="德劳帕迪·穆尔穆", model_extra={})
        assert extract_gemma4_content(msg) == "德劳帕迪·穆尔穆"

    def test_reads_reasoning_content_when_content_empty(self):
        msg = SimpleNamespace(
            content="",
            reasoning_content="Thinking...\nFormulate the Final Answer (Concise):** Droupadi Murmu.",
            model_extra={},
        )
        assert extract_gemma4_content(msg) == "Droupadi Murmu."

    def test_reads_channel_tail(self):
        msg = SimpleNamespace(
            content="",
            reasoning_content="<|channel>thought\nlong reasoning<channel|>Canberra",
            model_extra={},
        )
        assert extract_gemma4_content(msg) == "Canberra"

    def test_reads_model_extra_reasoning(self):
        msg = SimpleNamespace(
            content=None,
            reasoning_content=None,
            model_extra={"reasoning_content": "Final Answer: Four"},
        )
        assert extract_gemma4_content(msg) == "Four"
