"""Deepseek API 连通性测试 + 简易调用示例。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.llm_client import LLMClient


def main() -> None:
    print("API_BASE:", CONFIG.api_base)
    print("MODEL:", CONFIG.model)
    print("KEY_LEN:", len(CONFIG.api_key))

    llm = LLMClient()
    out = llm.generate(
        "你是一位简洁的助手。",
        "用一句话介绍 RAG（Retrieval-Augmented Generation）。",
        max_tokens=120,
    )
    print("---ANSWER---")
    print(out)
    print("---USAGE---")
    print(llm.usage.to_dict())


if __name__ == "__main__":
    main()
