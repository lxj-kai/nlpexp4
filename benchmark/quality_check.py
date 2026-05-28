"""Step 4: Quality check - verify generated benchmark entries."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BENCHMARK_ROOT
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

CHECK_PROMPT = """你是质量检查员。检查以下 RAG 噪音鲁棒性测试数据的质量。

检查项：
1. 问题是否有意义、有难度
2. 正确答案是否能从 positive 文档中推导出来
3. 反事实文档是否足够逼真（只改了关键信息，其他部分保持真实）
4. 错误答案是否与反事实文档一致
5. negative 文档是否与问题相关但不包含答案

请给出 1-5 分的总评分和简短评价。
输出 JSON: {"score": N, "issues": "问题描述或'无'"}"""


def check_entry(entry: dict) -> dict:
    user_msg = json.dumps(entry, ensure_ascii=False, indent=2)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CHECK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"  [ERROR] {e}")
    return {"score": 0, "issues": "检查失败"}


def main():
    print("=== Step 4: Quality Check ===")
    bench_file = BENCHMARK_ROOT / "output" / "bench_zh.json"
    if not bench_file.exists():
        print("No benchmark file found. Run assemble.py first.")
        return

    records = []
    with open(bench_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = []
    total_score = 0
    for i, rec in enumerate(records[:20]):
        print(f"  Checking #{rec['id']} ({i+1}/{min(len(records), 20)})")
        check = {"id": rec["id"], "query": rec["query"][:50]}
        result = check_entry(rec)
        check.update(result)
        results.append(check)
        total_score += result.get("score", 0)
        print(f"    Score: {result.get('score', '?')} - {result.get('issues', '?')[:60]}")

    n = len(results)
    avg = total_score / n if n else 0
    print(f"\nAverage score: {avg:.1f}/5 ({n} entries checked)")

    report_file = BENCHMARK_ROOT / "output" / "quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({"avg_score": avg, "n_checked": n, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
