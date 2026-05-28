"""Quick test: regenerate 1 entry with new multi-doc format."""
import json
from generate import generate_for_topic

entries = generate_for_topic("history", ["三国时期", "唐朝"])
if entries:
    e = entries[0]
    print("Q:", e["question"])
    print("A:", e["answer"])
    print(f"Positive: {len(e['positive'])} docs")
    for i, p in enumerate(e["positive"]):
        print(f"  pos[{i}]: {p[:80]}...")
    print(f"Positive_wrong: {len(e['positive_wrong'])} docs")
    for i, pw in enumerate(e["positive_wrong"]):
        print(f"  pw[{i}]: {pw[:80]}...")
    print(f"Fake: {e['fakeanswer']}")
    print(f"Negative: {len(e['negative_candidates'])} docs")
else:
    print("No entries generated")
