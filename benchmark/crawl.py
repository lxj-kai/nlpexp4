"""Step 1: Crawl real documents as semantic noise sources.

Uses Bing/Baidu search or Wikipedia API to fetch topic-related documents.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from config import TOPICS, BENCHMARK_ROOT

CRAWL_DIR = BENCHMARK_ROOT / "crawl_cache"
CRAWL_DIR.mkdir(exist_ok=True)

WIKI_API = "https://zh.wikipedia.org/w/api.php"


def search_wikipedia(keyword: str, limit: int = 15) -> list[dict]:
    """Search Chinese Wikipedia for articles related to keyword."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "srlimit": limit,
        "utf8": 1,
        "format": "json",
    })
    url = f"{WIKI_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "NLPExp4-Benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = data.get("query", {}).get("search", [])
        return [{"title": r["title"], "snippet": r.get("snippet", "")} for r in results]
    except Exception as e:
        print(f"  [WARN] Wikipedia search failed for '{keyword}': {e}")
        return []


def fetch_wikipedia_content(title: str, max_chars: int = 2000) -> str:
    """Fetch plain text extract of a Wikipedia article."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": 0,
        "explaintext": 1,
        "exchars": max_chars,
        "format": "json",
    })
    url = f"{WIKI_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "NLPExp4-Benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except Exception as e:
        print(f"  [WARN] Wikipedia fetch failed for '{title}': {e}")
    return ""


def crawl_topic(topic: str, keywords: list[str]) -> list[dict]:
    """Crawl documents for a topic using its keywords."""
    cache_file = CRAWL_DIR / f"{topic}.json"
    if cache_file.exists():
        print(f"  [CACHE] {topic} already crawled")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    docs = []
    seen_titles = set()

    for kw in keywords:
        print(f"  Searching: {kw}")
        results = search_wikipedia(kw)
        for r in results:
            if r["title"] in seen_titles:
                continue
            seen_titles.add(r["title"])
            content = fetch_wikipedia_content(r["title"])
            if content and len(content) > 100:
                docs.append({
                    "title": r["title"],
                    "text": content.strip(),
                    "keyword": kw,
                })
            time.sleep(0.5)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  [{topic}] Crawled {len(docs)} documents")
    return docs


def main():
    print("=== Step 1: Crawling documents ===")
    total = 0
    for topic, keywords in TOPICS.items():
        print(f"\n[{topic}]")
        docs = crawl_topic(topic, keywords)
        total += len(docs)
    print(f"\nTotal: {total} documents crawled")


if __name__ == "__main__":
    main()
