"""StrategyQA (NoiserBench) boolean gold answer handling."""
from __future__ import annotations

from src.noiser_loader import iter_noiser_records
from src.prompts import build_judge_prompt, build_naive_prompt


def test_strategyqa_gold_answers_not_empty():
    recs = list(iter_noiser_records("strategyqa"))
    assert len(recs) == 500
    empty = [r for r in recs if not r.answers_norm]
    assert empty == [], f"{len(empty)} records still missing gold"
    assert all(a in ("yes", "no") for r in recs for a in r.answers_norm)


def test_strategyqa_false_gold_is_no():
    rec = next(r for r in iter_noiser_records("strategyqa") if r.answers_norm == ["no"])
    assert rec.answers_norm == ["no"]


def test_strategyqa_prompts_yes_no():
    sys_msg, user = build_naive_prompt(
        "Is the sky blue?",
        ["doc"],
        language="en",
        dataset="noiser_bench",
        subset="strategyqa",
    )
    assert "Yes or No" in sys_msg or "yes/no" in sys_msg.lower()
    assert user

    judge_sys, _ = build_judge_prompt(
        "Is the sky blue?",
        "Yes",
        ["yes"],
        language="en",
        dataset="noiser_bench",
        subset="strategyqa",
    )
    assert "yes/no" in judge_sys.lower()
