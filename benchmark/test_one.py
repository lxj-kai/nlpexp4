"""Quick test: crawl + generate 1 entry."""
import json
from crawl import crawl_topic
from generate import generate_for_topic

docs = crawl_topic("history", ["三国时期", "唐朝"])
print(f"Crawled {len(docs)} docs")
if docs:
    title = docs[0]["title"]
    tlen = len(docs[0]["text"])
    print(f"First: {title} ({tlen} chars)")

entries = generate_for_topic("history", ["三国时期", "唐朝"])
print(f"Generated {len(entries)} entries")
if entries:
    e = entries[0]
    print(json.dumps({"q": e["question"], "a": e["answer"], "fake": e["fakeanswer"]}, ensure_ascii=False, indent=2))
