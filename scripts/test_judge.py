"""DeepSeek Judge 连通性测试（生成仍走 LM Studio）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.evaluator import Evaluator
from src.llm_client import LLMClient, get_judge_client
from src.rag_pipeline import RAGResult


def main() -> None:
    print("GEN:", CONFIG.model, "@", CONFIG.api_base)
    print("JUDGE:", CONFIG.judge_model, "@", CONFIG.judge_api_base)
    print("JUDGE_KEY_LEN:", len(CONFIG.judge_api_key))

    llm = LLMClient()
    judge = get_judge_client()
    ev = Evaluator(llm=llm, judge_llm=judge, use_semantic_attribution=False)

    sample = RAGResult(
        sample_id=0,
        query="2021年北京大兴机场年旅客吞吐量是多少？",
        gold_answers=["2500万人次"],
        prediction="2500万人次",
        docs=["大兴机场2021年旅客吞吐量突破2500万人次"],
        labels=["positive"],
        noise_ratio=0.0,
        noise_type="semantic",
        noise_position="interleave",
        metadata={},
    )
    m = ev.evaluate_one(sample, language="zh")
    print("judge_score:", m.judge_score)
    print("judge_correct:", m.judge_correct)


if __name__ == "__main__":
    main()
