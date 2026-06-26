"""位置子实验结果摘要。"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--pattern", default="exp1_noise_impact_*_position_*")
args = p.parse_args()
files = sorted((ROOT / "experiments" / "results").glob(f"{args.pattern}.json"))
if not files:
    print("no position results")
    sys.exit(0)
data = json.loads(files[-1].read_text(encoding="utf-8"))
print(f"file: {files[-1].name}\n")
print("| position    |    F1 | ROUGE-L | contains |   ISR |   NAR |")
print("|:------------|------:|--------:|---------:|------:|------:|")
for r in data["results"]:
    c = r["condition"]
    s = r["summary"]
    row = "| {:<11} | {:.3f} | {:.3f}   | {:.3f}    | {:.3f} | {:.3f} |".format(
        c["noise_position"],
        s.get("token_f1") or 0,
        s.get("rouge_l") or 0,
        s.get("contains") or 0,
        s.get("isr") or 0,
        s.get("nar") or 0,
    )
    print(row)
