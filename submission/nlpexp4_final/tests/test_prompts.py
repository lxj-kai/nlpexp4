"""Unit tests for dataset-specific naive prompts."""
from __future__ import annotations

from src.prompts import build_naive_prompt, get_naive_profile


class TestDatasetPromptProfiles:
    def test_rgb_keeps_short_answer_instruction(self):
        _, user = build_naive_prompt("q", ["doc"], language="zh", dataset="rgb")
        assert "一句话" in user

    def test_cmedqa_requires_paragraph_advice(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="zh", dataset="cmedqa")
        assert "医学" in sys_msg or "患者" in sys_msg
        assert "2–5 句" in user
        assert "答案尽量简短" not in user

    def test_miriad_english_clinical_style(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="en", dataset="miriad")
        assert "clinical" in sys_msg.lower()
        assert "2–5 sentences" in user

    def test_2wiki_keeps_short_factoid(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="en", dataset="2wiki")
        assert "multi-hop" in sys_msg.lower() or "factual" in sys_msg.lower()
        assert "short phrase" in user.lower()

    def test_bright_long_form_generation(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="en", dataset="bright")
        assert "long-form" in sys_msg.lower() or "reasoning" in sys_msg.lower()
        assert "complete paragraph" in user.lower()
        assert "short phrase" not in user.lower()

    def test_bright_truncates_long_query(self):
        long_query = "x" * 10000
        _, user = build_naive_prompt(long_query, ["doc"], language="en", dataset="bright")
        assert "truncated for context limit" in user
        assert "x" * 10000 not in user

    def test_multihop_rag_short_factoid(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="en", dataset="multihop_rag")
        assert "news" in sys_msg.lower() or "multi-hop" in sys_msg.lower()
        assert "short phrase" in user.lower()

    def test_tempo_forum_paragraph(self):
        sys_msg, user = build_naive_prompt("q", ["doc"], language="en", dataset="tempo")
        assert "forum" in sys_msg.lower()
        assert "2–5 sentences" in user or "complete paragraph" in user.lower()

    def test_bright_judge_long_form(self):
        from src.prompts import build_judge_prompt

        sys_msg, _ = build_judge_prompt("q", "pred", ["gold"], language="en", dataset="bright")
        assert "long-form" in sys_msg.lower() or "reasoning" in sys_msg.lower()

    def test_multihop_judge_short_entity(self):
        from src.prompts import build_judge_prompt

        sys_msg, _ = build_judge_prompt("q", "pred", ["gold"], language="en", dataset="multihop_rag")
        assert "news" in sys_msg.lower() or "entity" in sys_msg.lower()

    def test_unknown_dataset_falls_back_to_rgb(self):
        profile = get_naive_profile("unknown")
        assert profile is get_naive_profile("rgb")
