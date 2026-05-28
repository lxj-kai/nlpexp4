"""Step 3: Assemble generated entries into RGB-compatible JSONL format."""
from __future__ import annotations

import json
from pathlib import Path

from config import TOPICS, BENCHMARK_ROOT

GEN_DIR = BENCHMARK_ROOT / "gen_cache"
OUTPUT_DIR = BENCHMARK_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def assemble():
    print("=== Step 3: Assembling benchmark ===")
    records = []
    record_id = 2001

    for topic in TOPICS:
        gen_file = GEN_DIR / f"{topic}.json"
        if not gen_file.exists():
            print(f"  [SKIP] {topic} not generated")
            continue

        with open(gen_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            negatives = entry.get("negative_candidates", [])[:5]
            if len(negatives) < 2:
                continue

            positives = entry.get("positive", [])
            if isinstance(positives, str):
                positives = [positives]

            pw = entry.get("positive_wrong", [])
            if isinstance(pw, str):
                pw = [pw]

            record = {
                "id": record_id,
                "query": entry["question"],
                "answer": entry["answer"],
                "positive": positives,
                "negative": negatives,
                "positive_wrong": pw,
                "fakeanswer": entry.get("fakeanswer", ""),
            }
            records.append(record)
            record_id += 1

    output_file = OUTPUT_DIR / "bench_zh.json"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Assembled {len(records)} records -> {output_file}")

    stats_file = OUTPUT_DIR / "stats.json"
    topic_counts = {}
    for topic in TOPICS:
        gen_file = GEN_DIR / f"{topic}.json"
        if gen_file.exists():
            with open(gen_file, "r", encoding="utf-8") as f:
                topic_counts[topic] = len(json.load(f))

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_records": len(records),
            "topics": topic_counts,
        }, f, ensure_ascii=False, indent=2)

    return records


if __name__ == "__main__":
    assemble()
