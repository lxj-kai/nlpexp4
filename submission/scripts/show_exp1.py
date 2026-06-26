"""打印 exp1 最新结果的汇总表。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pattern", default="exp1_noise_impact_*")
args = parser.parse_args()
files = sorted((ROOT / "experiments" / "results").glob(f"{args.pattern}.json"))
if not files:
    print("no exp1 results yet")
    sys.exit(0)

data = json.loads(files[-1].read_text(encoding="utf-8"))
print(f"file: {files[-1].name}")
print(f"experiment: {data['experiment']}")
print(f"n_results: {len(data['results'])}\n")

print("| ratio | type           |  n |    F1 | ROUGE-L | contains |   ISR |   NAR |")
print("|------:|:---------------|---:|------:|--------:|---------:|------:|------:|")
for r in data["results"]:
    c = r["condition"]
    s = r["summary"]
    row = "| {:.2f} | {:<14} | {:2d} | {:.3f} | {:.3f}   | {:.3f}    | {:.3f} | {:.3f} |".format(
        c["noise_ratio"],
        c["noise_type"],
        s.get("n", 0),
        s.get("token_f1") or 0,
        s.get("rouge_l") or 0,
        s.get("contains") or 0,
        s.get("isr") or 0,
        s.get("nar") or 0,
    )
    print(row)

print("\nrobustness_table:")
for r in data["robustness_table"]:
    print(" ", r)
