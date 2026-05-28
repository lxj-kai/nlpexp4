"""Benchmark data engine API endpoints."""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmark"
CRAWL_DIR = BENCHMARK_ROOT / "crawl_cache"
GEN_DIR = BENCHMARK_ROOT / "gen_cache"
OUTPUT_DIR = BENCHMARK_ROOT / "output"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WIKI_API = "https://zh.wikipedia.org/w/api.php"


@router.get("/topics")
def list_topics():
    topics = {}
    for f in sorted(CRAWL_DIR.glob("*.json")):
        topic = f.stem
        with open(f, "r", encoding="utf-8") as fh:
            docs = json.load(fh)
        gen_file = GEN_DIR / f"{topic}.json"
        entries = []
        if gen_file.exists():
            with open(gen_file, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        topics[topic] = {
            "crawled_docs": len(docs),
            "generated_entries": len(entries),
        }
    return {"topics": topics}


@router.get("/topic/{topic}")
def get_topic_detail(topic: str):
    crawl_file = CRAWL_DIR / f"{topic}.json"
    if not crawl_file.exists():
        raise HTTPException(404, f"Topic '{topic}' not found")

    with open(crawl_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    gen_file = GEN_DIR / f"{topic}.json"
    entries = []
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

    return {
        "topic": topic,
        "crawled_docs": [{"title": d["title"], "text": d["text"][:300], "keyword": d.get("keyword", "")} for d in docs],
        "entries": entries,
    }


@router.get("/stats")
def get_stats():
    stats_file = OUTPUT_DIR / "stats.json"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_records": 0, "topics": {}}


class GenerateRequest(BaseModel):
    keyword: str = "量子力学"


def _wiki_search(keyword: str, limit: int = 8) -> list[dict]:
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": keyword,
        "srlimit": limit, "utf8": 1, "format": "json",
    })
    req = urllib.request.Request(f"{WIKI_API}?{params}", headers={"User-Agent": "NLPExp4/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("query", {}).get("search", [])
    except Exception:
        return []


def _wiki_extract(title: str, max_chars: int = 1500) -> str:
    params = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "extracts",
        "exintro": 0, "explaintext": 1, "exchars": max_chars, "format": "json",
    })
    req = urllib.request.Request(f"{WIKI_API}?{params}", headers={"User-Agent": "NLPExp4/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for page in data.get("query", {}).get("pages", {}).values():
            return page.get("extract", "")
    except Exception:
        pass
    return ""


def _llm_generate(positive_text: str) -> dict | None:
    from src.llm_client import LLMClient
    llm = LLMClient()

    prompt = f"""基于以下文档，生成一条测试数据。

文档：
{positive_text[:800]}

请生成：
1. 一个有难度的中文问题
2. 正确答案（简洁）
3. 3 篇支撑文档（每篇 80-120 字）
4. 4 篇反事实文档（保持风格一致，每篇修改不同的关键信息，80-120 字）

严格JSON，不要其他文字：
{{"question":"...", "answer":"...", "positive_docs":["p1","p2","p3"], "positive_wrong_docs":["w1","w2","w3","w4"]}}"""

    text = llm.generate(
        system="RAG benchmark 数据构造专家。只输出JSON。",
        user=prompt,
        temperature=0.5,
        max_tokens=1500,
    )
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


from fastapi.responses import StreamingResponse


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate_one")
def generate_one_entry(req: GenerateRequest):
    def event_stream():
        yield _sse_event({"step": 1, "title": "搜索维基百科", "status": "running", "data": None})

        all_titles = set()
        search_terms = [req.keyword]
        parts = req.keyword.split()
        if len(parts) > 1:
            search_terms.extend(parts[:3])
        search_terms.append(req.keyword + " 原理")
        search_terms.append(req.keyword + " 历史")

        for term in search_terms:
            results = _wiki_search(term, limit=10)
            for r in results:
                all_titles.add(r["title"])
            time.sleep(0.15)
        titles = list(all_titles)[:20]
        yield _sse_event({"step": 1, "title": "搜索维基百科", "status": "done",
                          "data": {"keyword": req.keyword, "found": len(titles), "titles": titles}})

        yield _sse_event({"step": 2, "title": "获取文档内容", "status": "running", "data": None})
        docs = []
        for t in titles[:12]:
            text = _wiki_extract(t)
            if text and len(text) > 100:
                docs.append({"title": t, "text": text})
            time.sleep(0.15)
        yield _sse_event({"step": 2, "title": "获取文档内容", "status": "done",
                          "data": {"fetched": len(docs),
                                   "docs": [{"title": d["title"], "text": d["text"][:200]} for d in docs]}})

        if not docs:
            yield _sse_event({"step": 0, "title": "error", "status": "failed", "data": {"error": "未获取到足够的文档"}})
            return

        yield _sse_event({"step": 3, "title": "LLM 生成测试数据", "status": "running", "data": None})
        source_doc = docs[0]
        parsed = _llm_generate(source_doc["text"])

        if not parsed:
            yield _sse_event({"step": 3, "title": "LLM 生成测试数据", "status": "failed", "data": None})
            yield _sse_event({"step": 0, "title": "error", "status": "failed", "data": {"error": "LLM 生成失败"}})
            return

        yield _sse_event({"step": 3, "title": "LLM 生成测试数据", "status": "done",
                          "data": {"question": parsed.get("question", ""), "answer": parsed.get("answer", "")}})

        yield _sse_event({"step": 4, "title": "组装 RGB 格式", "status": "running", "data": None})

        pos_docs = parsed.get("positive_docs", [])
        if isinstance(pos_docs, str):
            pos_docs = [pos_docs]
        pos_docs = [source_doc["text"][:600]] + pos_docs[:5]

        pw_docs = parsed.get("positive_wrong_docs", [])
        if isinstance(pw_docs, str):
            pw_docs = [pw_docs]

        negatives = [d["text"][:500] for d in docs[1:]]

        entry = {
            "id": int(time.time()) % 100000,
            "query": parsed["question"],
            "answer": parsed["answer"],
            "positive": pos_docs,
            "negative": negatives,
            "positive_wrong": pw_docs[:4],
            "source": source_doc["title"],
            "keyword": req.keyword,
        }

        yield _sse_event({"step": 4, "title": "组装 RGB 格式", "status": "done",
                          "data": {"n_positive": len(pos_docs), "n_negative": len(negatives), "n_wrong": len(entry["positive_wrong"])}})

        yield _sse_event({"step": 5, "title": "complete", "status": "done", "data": {"entry": entry}})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
