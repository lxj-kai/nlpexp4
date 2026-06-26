"""汇总 exp_noise_gradient 结果：按数据集 × 方法 × 噪音比例打印 judge 正确率。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.results_paths import glob_results


def load_summaries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = (
        data.get("dataset_key")
        or data.get("extras", {}).get("dataset_key")
        or data.get("experiment", path.stem)
    )
    rows: list[dict] = []
    for block in data.get("results", []):
        cond = block.get("condition", {})
        summ = block.get("summary", {})
        if cond.get("noise_ratio", -1) < 0:
            continue
        rows.append(
            {
                "dataset": key,
                "method": cond.get("method", "?"),
                "ratio": cond.get("noise_ratio"),
                "n": summ.get("n", block.get("n", 0)),
                "judge_score": summ.get("judge_score"),
                "judge_correct": summ.get("judge_correct"),
                "isr": summ.get("isr"),
                "nar": summ.get("nar"),
            }
        )
    return rows


def print_table(all_rows: list[dict]) -> None:
    datasets = sorted({r["dataset"] for r in all_rows})
    methods = sorted({r["method"] for r in all_rows})
    ratios = sorted({r["ratio"] for r in all_rows})

    print("\n=== Judge 正确率 (judge_correct) ===")
    header = f"{'dataset':<10} {'method':<12}" + "".join(f" r={r:<4}" for r in ratios)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        for m in methods:
            cells = [f"{ds:<10} {m:<12}"]
            for ratio in ratios:
                hit = next(
                    (r for r in all_rows if r["dataset"] == ds and r["method"] == m and r["ratio"] == ratio),
                    None,
                )
                if hit and hit.get("judge_correct") is not None:
                    cells.append(f" {hit['judge_correct']*100:5.1f}%")
                elif hit and hit.get("judge_score") is not None:
                    cells.append(f" {hit['judge_score']*100:5.1f}s")
                else:
                    cells.append("   n/a")
            print("".join(cells))

    print("\n=== Judge 分数 (judge_score, 0-1) ===")
    for ds in datasets:
        print(f"\n[{ds}]")
        for m in methods:
            parts = []
            for ratio in ratios:
                hit = next(
                    (r for r in all_rows if r["dataset"] == ds and r["method"] == m and r["ratio"] == ratio),
                    None,
                )
                if hit and hit.get("judge_score") is not None:
                    parts.append(f"r={ratio}:{hit['judge_score']:.3f}")
                else:
                    parts.append(f"r={ratio}:n/a")
            print(f"  {m:<12} " + " | ".join(parts))

    # naive 恢复率：高噪音 prompt/confidence vs naive
    print("\n=== 矫正恢复 (judge_correct@0.75 - naive@0.75) ===")
    for ds in datasets:
        naive_h = next(
            (r for r in all_rows if r["dataset"] == ds and r["method"] == "naive" and r["ratio"] == 0.75),
            None,
        )
        if not naive_h or naive_h.get("judge_correct") is None:
            continue
        base = naive_h["judge_correct"]
        for m in ("prompt", "confidence"):
            hit = next(
                (r for r in all_rows if r["dataset"] == ds and r["method"] == m and r["ratio"] == 0.75),
                None,
            )
            if hit and hit.get("judge_correct") is not None:
                delta = hit["judge_correct"] - base
                print(f"  {ds} {m}: {delta:+.3f} (naive={base:.3f} -> {hit['judge_correct']:.3f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_files", nargs="*", help="结果 JSON；留空则取 results 下最新 exp_noise_gradient_*")
    args = p.parse_args()

    paths: list[Path] = []
    if args.json_files:
        paths = [Path(x) for x in args.json_files]
    else:
        paths = glob_results("exp_noise_gradient_*_n*.json")
        # 每个 dataset 只取最新
        latest: dict[str, Path] = {}
        for pth in paths:
            key = pth.stem.split("_n")[0]
            latest[key] = pth
        paths = list(latest.values())

    if not paths:
        print("no result files found")
        sys.exit(1)

    all_rows: list[dict] = []
    for pth in paths:
        all_rows.extend(load_summaries(pth))
        print(f"loaded {pth.name}")

    print_table(all_rows)


if __name__ == "__main__":
    main()
