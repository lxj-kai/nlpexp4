"""汇总 closed-book vs Oracle RAG 对照结果（DeepSeek Judge）。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

import sys

sys.path.insert(0, str(ROOT))
from src.results_paths import glob_results


def _latest(pattern: str) -> Path | None:
    files = sorted(glob_results(pattern))
    return files[-1] if files else None


def main() -> None:
    rows: list[dict] = []
    for dataset in ("cmedqa", "miriad", "2wiki"):
        fp = _latest(f"exp_closed_book_{dataset}_*.json")
        if not fp:
            print(f"missing results for {dataset}")
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        comp = data.get("comparison") or {}
        args = data.get("args") or {}
        closed = comp.get("closed_book") or {}
        rag = comp.get("oracle_rag_naive_r0") or {}
        rows.append(
            {
                "dataset": dataset,
                "language": args.get("language"),
                "n": args.get("n"),
                "closed_judge": closed.get("judge_score"),
                "rag_judge": rag.get("judge_score"),
                "delta_judge": comp.get("delta_judge_rag_minus_closed"),
                "closed_acc": closed.get("judge_correct"),
                "rag_acc": rag.get("judge_correct"),
                "file": fp.name,
            }
        )

    print("| 数据集 | 语言 | n | Closed Judge | Oracle RAG Judge | ΔJudge | Closed Acc | RAG Acc |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['dataset']} | {r['language']} | {r['n']} | "
            f"{r['closed_judge']:.4f} | {r['rag_judge']:.4f} | {r['delta_judge']:+.4f} | "
            f"{r['closed_acc']:.4f} | {r['rag_acc']:.4f} |"
        )


if __name__ == "__main__":
    main()
