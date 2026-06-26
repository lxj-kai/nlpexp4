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

from backend.deps import get_records

WIKI_API = "https://zh.wikipedia.org/w/api.php"

NOISER_BENCH_SUBSETS = {
    "2wikimqa",
    "2wikiimqa",
    "bamboogle",
    "hotpotqa",
    "nq",
    "priorqa",
    "rgb_nb",
    "strategyqa",
    "tempqa",
}

SUBSET_OPERATIONS = {
    "main": "qa",
    "refine": "refine",
    "fact": "counterfactual",
    "int": "integration",
}


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


def _sort_key(record) -> tuple[int, int | str]:
    raw_id = getattr(record, "id", "")
    if isinstance(raw_id, int):
        return (0, raw_id)
    if isinstance(raw_id, str) and raw_id.isdigit():
        return (0, int(raw_id))
    return (1, str(raw_id))


def _record_operation(record, subset: str) -> str:
    meta = getattr(record, "meta", None) or {}
    if meta.get("operation"):
        return str(meta["operation"])
    if subset in NOISER_BENCH_SUBSETS:
        return "noiserbench"
    return SUBSET_OPERATIONS.get(subset, "qa")


def _record_scope_type(record) -> str:
    meta = getattr(record, "meta", None) or {}
    if meta.get("scope_type"):
        return str(meta["scope_type"])
    if meta.get("support_path"):
        return "support_path"
    if getattr(record, "positive_wrong", None):
        return "counterfactual"
    return "standard"


def _record_scope_desc(record, subset: str) -> str:
    meta = getattr(record, "meta", None) or {}
    if meta.get("scope_desc"):
        return str(meta["scope_desc"])
    counts = [
        f"P{len(getattr(record, 'positive', []) or [])}",
        f"N{len(getattr(record, 'negative', []) or [])}",
    ]
    wrong_count = len(getattr(record, "positive_wrong", []) or [])
    if wrong_count:
        counts.append(f"CF{wrong_count}")
    return f"{subset} · " + " / ".join(counts)


def _record_to_benchmark_dict(record, subset: str) -> dict:
    meta = getattr(record, "meta", None) or {}
    return {
        "id": record.id,
        "query": record.query,
        "answer": record.answer,
        "positive": record.positive,
        "negative": record.negative,
        "positive_wrong": record.positive_wrong,
        "fakeanswer": record.fakeanswer,
        "mobilemem_meta": meta,
        "benchmark_meta": {
            "operation": _record_operation(record, subset),
            "scope_type": _record_scope_type(record),
            "scope_desc": _record_scope_desc(record, subset),
            "metric_name": str(meta.get("metric_name") or ""),
            "has_support_path": bool(meta.get("support_path")),
        },
        "language": getattr(record, "language", "zh"),
        "subset": getattr(record, "subset", subset),
    }


def _length_set(records: list, attr: str) -> list[int]:
    return sorted({len(getattr(record, attr, []) or []) for record in records})


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _dataset_response(language: str, subset: str) -> dict:
    try:
        records = sorted(get_records(language, subset), key=_sort_key)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(404, f"Dataset not found: {language}/{subset}") from e

    benchmark_records = [_record_to_benchmark_dict(record, subset) for record in records]
    return {
        "dataset": subset,
        "language": language,
        "filename": f"{language}/{subset}",
        "total": len(records),
        "summary": {
            "id_range": [records[0].id, records[-1].id] if records else [],
            "positive_docs": _length_set(records, "positive"),
            "negative_docs": _length_set(records, "negative"),
            "positive_wrong_docs": _length_set(records, "positive_wrong"),
            "answer_count": _length_set(records, "answer"),
            "operations": _count_values([r["benchmark_meta"]["operation"] for r in benchmark_records]),
            "scope_types": _count_values([r["benchmark_meta"]["scope_type"] for r in benchmark_records]),
            "has_support_path": any(r["benchmark_meta"]["has_support_path"] for r in benchmark_records),
        },
        "records": benchmark_records,
    }


@router.get("/dataset/{subset}")
def get_benchmark_dataset(subset: str, language: str = "zh"):
    return _dataset_response(language, subset)


@router.get("/mobilemem_shopping_graph_hard_120")
def get_mobilemem_shopping_graph_hard_120():
    return _dataset_response("zh", "mobilemem_shopping_graph_hard_120")


@router.get("/mobilemem_shopping_graph_noncalc_hard_120")
def get_mobilemem_shopping_graph_noncalc_hard_120():
    return _dataset_response("zh", "mobilemem_shopping_graph_noncalc_hard_120")


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
