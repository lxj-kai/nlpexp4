"""打印 exp4（vs 现有方法）汇总。"""
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
from src.visualize import plot_robustness_radar  # noqa: E402

files = sorted((ROOT / "experiments" / "results").glob("exp4_existing_methods_zh_*.json"))
if not files:
    print("no exp4 results")
    sys.exit(0)
latest = files[-1]
data = json.loads(latest.read_text(encoding="utf-8"))
print(f"file: {latest.name}\n")

print("| method     |  n |    F1 | ROUGE-L | contains |   ISR |   NAR | API calls |")
print("|:-----------|---:|------:|--------:|---------:|------:|------:|----------:|")
for r in data["results"]:
    c = r["condition"]
    s = r["summary"]
    api_calls = sum(row.get("rows_meta", {}).get("api_calls", 0) for row in r.get("rows", []))
    avg_calls = (
        sum(row.get("metadata", {}).get("api_calls", 1) for row in r.get("rows", [])) / max(1, len(r.get("rows", [])))
        if r.get("rows")
        else 1
    )
    print(
        "| {:<10} | {:2d} | {:.3f} | {:.3f}   | {:.3f}    | {:.3f} | {:.3f} |".format(
            c["method"],
            s.get("n", 0),
            s.get("token_f1") or 0,
            s.get("rouge_l") or 0,
            s.get("contains") or 0,
            s.get("isr") or 0,
            s.get("nar") or 0,
        )
    )

tbl = data.get("robustness_table", [])
if tbl:
    out = plot_robustness_radar(
        tbl,
        out_path=ROOT / "figures" / "exp4_robustness_radar.png",
    )
    print(f"\nfigure: {out}")
