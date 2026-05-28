"""Step 2: Use LLM to generate Q/A/positive/counterfactual from crawled docs.

Takes crawled documents and generates benchmark entries using DeepSeek API.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TOPICS, TARGET_PER_TOPIC, BENCHMARK_ROOT
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
)
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

GEN_DIR = BENCHMARK_ROOT / "gen_cache"
GEN_DIR.mkdir(exist_ok=True)

GENERATE_PROMPT = """你是一个高质量 RAG 噪音鲁棒性 benchmark 的数据构造专家。

给你一篇真实文档，请完成以下任务：

1. 基于文档内容，提出一个**有难度的中文问题**（需要综合理解才能回答，不能太简单）
2. 给出**正确答案**（简洁、明确）
3. 生成 **3 篇正确支撑文档（positive）**：每篇从不同角度描述相关内容，200-300字，都能支持正确答案
4. 生成 **2 篇反事实文档（positive_wrong）**：
   - 每篇保持整体结构和语言风格与原文一致
   - 每篇修改不同的关键事实（如人名、年份、数字、地点）
   - 修改后的内容要看起来合理，难以被模型识别为错误
   - 两篇分别引导到不同的错误答案
5. 给出 2 个对应的**错误答案**

请严格以JSON格式输出：
{
  "question": "问题文本",
  "answer": "正确答案",
  "positive_docs": ["支撑文档1", "支撑文档2", "支撑文档3"],
  "positive_wrong_docs": ["反事实文档1", "反事实文档2"],
  "fakeanswers": ["错误答案1", "错误答案2"]
}

注意：
- 问题要有深度，不能是简单的"是什么"型问题
- 反事实文档要高度逼真，只改关键信息
- 3篇正确文档应从不同角度阐述，不要简单复制
- 确保 JSON 格式正确"""


def call_llm(prompt: str, system: str = GENERATE_PROMPT, max_tokens: int = 1024) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [RETRY {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return ""


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def generate_for_topic(topic: str, keywords: list[str]) -> list[dict]:
    cache_file = GEN_DIR / f"{topic}.json"
    if cache_file.exists():
        print(f"  [CACHE] {topic} already generated")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    crawl_file = BENCHMARK_ROOT / "crawl_cache" / f"{topic}.json"
    if not crawl_file.exists():
        print(f"  [SKIP] No crawled docs for {topic}")
        return []

    with open(crawl_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if len(docs) < 3:
        print(f"  [SKIP] Not enough docs for {topic} ({len(docs)})")
        return []

    entries = []
    used_docs = set()

    for doc in docs[:TARGET_PER_TOPIC * 2]:
        if len(entries) >= TARGET_PER_TOPIC:
            break
        if doc["title"] in used_docs or len(doc["text"]) < 150:
            continue

        print(f"  Generating from: {doc['title']}")
        user_msg = f"文档标题：{doc['title']}\n\n文档内容：\n{doc['text'][:1500]}"
        raw = call_llm(user_msg)
        parsed = parse_json_response(raw)

        required = ("question", "answer", "positive_docs", "positive_wrong_docs", "fakeanswers")
        if not parsed or not all(k in parsed for k in required):
            print(f"    [FAIL] Bad response")
            continue

        other_docs = [d["text"] for d in docs if d["title"] != doc["title"] and d["title"] not in used_docs]

        pos_docs = parsed.get("positive_docs", [])
        if isinstance(pos_docs, str):
            pos_docs = [pos_docs]
        pos_docs = [doc["text"][:600]] + pos_docs[:3]

        pw_docs = parsed.get("positive_wrong_docs", [])
        if isinstance(pw_docs, str):
            pw_docs = [pw_docs]

        fakes = parsed.get("fakeanswers", [])
        if isinstance(fakes, str):
            fakes = [fakes]

        entry = {
            "question": parsed["question"],
            "answer": parsed["answer"],
            "positive": pos_docs,
            "positive_wrong": pw_docs[:3],
            "fakeanswer": fakes[0] if fakes else "",
            "negative_candidates": other_docs[:8],
            "topic": topic,
            "source_title": doc["title"],
        }
        entries.append(entry)
        used_docs.add(doc["title"])
        time.sleep(1)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"  [{topic}] Generated {len(entries)} entries")
    return entries


def main():
    print("=== Step 2: Generating benchmark entries ===")
    total = 0
    for topic, keywords in TOPICS.items():
        print(f"\n[{topic}]")
        entries = generate_for_topic(topic, keywords)
        total += len(entries)
    print(f"\nTotal: {total} entries generated")


if __name__ == "__main__":
    main()
