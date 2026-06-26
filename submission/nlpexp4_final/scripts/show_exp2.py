"""打印 exp2（矫正方法对比）汇总表 + 出图。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from src.visualize import plot_correction_compare  # noqa: E402

files = sorted((ROOT / "experiments" / "results").glob("exp2_correction_zh_*.json"))
if not files:
    print("no exp2 results")
    sys.exit(0)

latest = files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
print(f"file: {latest.name}\n")

print("| method     | ratio |  n |    F1 | ROUGE-L | contains |   ISR |   NAR |")
print("|:-----------|------:|---:|------:|--------:|---------:|------:|------:|")
for r in data["results"]:
    c = r["condition"]
    s = r["summary"]
    row = "| {:<10} | {:.2f}  | {:2d} | {:.3f} | {:.3f}   | {:.3f}    | {:.3f} | {:.3f} |".format(
        c["method"],
        c["noise_ratio"],
        s.get("n", 0),
        s.get("token_f1") or 0,
        s.get("rouge_l") or 0,
        s.get("contains") or 0,
        s.get("isr") or 0,
        s.get("nar") or 0,
    )
    print(row)

out = plot_correction_compare(str(latest))
print(f"\nfigure: {out}")
