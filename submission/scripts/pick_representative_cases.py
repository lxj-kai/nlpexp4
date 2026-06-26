"""从 exp_noise_gradient 结果中挑选有代表性的案例。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.results_paths import glob_results


def load_latest_results(results_dir: Path, dataset_key: str) -> dict | None:
    files = sorted(glob_results(f"exp_noise_gradient_{dataset_key}_n*.json"), reverse=True)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("extras", {}).get("dataset_key") == dataset_key or dataset_key in f.name:
            return data
    return None


def flatten_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for block in data.get("results", []):
        cond = block.get("condition", {})
        method = cond.get("method", "naive")
        ratio = cond.get("noise_ratio", 0.0)
        for row in block.get("rows", []):
            rows.append({**row, "method": method, "noise_ratio": ratio})
    return rows


def pick_cases(rows: list[dict], *, dataset: str) -> list[tuple[str, dict]]:
    """按类别各选 1-2 条代表。"""
    picked: list[tuple[str, dict]] = []
    used_ids: set[tuple] = set()

    def take(tag: str, row: dict | None) -> None:
        if row is None:
            return
        key = (row.get("sample_id"), row.get("noise_ratio"))
        if key in used_ids:
            return
        used_ids.add(key)
        picked.append((tag, row))

    naive_rows = [r for r in rows if r.get("method") == "naive"]

    # 1) 无噪音答对
    for r in sorted(naive_rows, key=lambda x: -(x.get("judge_score") or 0)):
        if r.get("noise_ratio") == 0.0 and r.get("judge_correct") == 1.0:
            take(f"{dataset}·无噪音·答对", r)
            break

    # 2) 无噪音答错（模型能力上限）
    for r in naive_rows:
        if r.get("noise_ratio") == 0.0 and r.get("judge_correct") == 0.0:
            take(f"{dataset}·无噪音·仍错", r)
            break

    # 3) 高噪音被带偏（NAR高 + 错）
    cands = [
        r
        for r in naive_rows
        if r.get("noise_ratio") == 0.75 and r.get("judge_correct") == 0.0 and (r.get("nar") or 0) >= 0.3
    ]
    cands.sort(key=lambda x: -(x.get("nar") or 0))
    if cands:
        take(f"{dataset}·高噪音·被带偏", cands[0])

    # 4) 高噪音仍答对（抗噪）
    for r in naive_rows:
        if r.get("noise_ratio") == 0.75 and r.get("judge_correct") == 1.0:
            take(f"{dataset}·高噪音·仍对", r)
            break

    # 5) 同题低→高噪音退化（找 sample_id 在 r=0 对 r=0.75 错）
    by_id: dict[int, dict[float, dict]] = {}
    for r in naive_rows:
        sid = r.get("sample_id")
        if sid is None:
            continue
        by_id.setdefault(sid, {})[float(r.get("noise_ratio", -1))] = r
    for sid, mp in by_id.items():
        r0, r75 = mp.get(0.0), mp.get(0.75)
        if not r0 or not r75:
            continue
        if r0.get("judge_correct") == 1.0 and r75.get("judge_correct") == 0.0:
            take(f"{dataset}·同题·低对高错 #{sid}", r75)
            picked.append((f"{dataset}·同题·低对高错 #{sid}·对照", r0))
            break

    # 6) 数值/短答案误判风险（预测与gold明显不同）
    for r in naive_rows:
        if r.get("noise_ratio", 0) < 0.5:
            continue
        gold = " / ".join(r.get("gold") or [])[:40]
        pred = (r.get("prediction") or "")[:40]
        if gold and pred and gold not in pred and pred not in gold:
            if r.get("judge_correct") == 0.0:
                take(f"{dataset}·Judge判错·明显不一致", r)
                break

    return picked


def fmt_row(tag: str, r: dict) -> str:
    gold = " / ".join(r.get("gold") or []) if isinstance(r.get("gold"), list) else str(r.get("gold", ""))
    q = (r.get("query") or "")[:70]
    pred = (r.get("prediction") or "")[:100]
    js = r.get("judge_score")
    js_s = f"{js:.2f}" if js is not None else "—"
    jc = "是" if r.get("judge_correct") == 1 else ("否" if r.get("judge_correct") == 0 else "—")
    return (
        f"\n### {tag}\n"
        f"- **noise**={r.get('noise_ratio')} · **Judge**={js_s} · **正确**={jc}\n"
        f"- **ISR**={r.get('isr', 0):.2f} · **NAR**={r.get('nar', 0):.2f}\n"
        f"- **问**：{q}{'…' if len(r.get('query',''))>70 else ''}\n"
        f"- **标**：{gold[:80]}{'…' if len(gold)>80 else ''}\n"
        f"- **答**：{pred}{'…' if len(r.get('prediction') or '')>100 else ''}\n"
    )


def summarize_dataset(rows: list[dict], dataset: str) -> str:
    naive = [r for r in rows if r.get("method") == "naive"]
    lines = [f"\n## {dataset} 汇总 (naive, n={len(set(r.get('sample_id') for r in naive))} 题)"]
    for ratio in (0.0, 0.5, 0.75):
        sub = [r for r in naive if r.get("noise_ratio") == ratio]
        if not sub:
            continue
        acc = sum(1 for r in sub if r.get("judge_correct") == 1.0) / len(sub)
        avg_js = sum(r.get("judge_score") or 0 for r in sub) / len(sub)
        avg_isr = sum(r.get("isr") or 0 for r in sub) / len(sub)
        avg_nar = sum(r.get("nar") or 0 for r in sub) / len(sub)
        lines.append(
            f"- r={ratio}: judge正确率 **{acc*100:.1f}%** · avg score **{avg_js:.2f}** · ISR **{avg_isr:.2f}** · NAR **{avg_nar:.2f}**"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--datasets", default="rgb,cmedqa,2wiki")
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    all_picked: list[tuple[str, dict]] = []
    summaries: list[str] = []

    for ds in args.datasets.split(","):
        ds = ds.strip()
        data = load_latest_results(results_dir, ds)
        if not data:
            print(f"未找到 {ds} 结果文件", file=sys.stderr)
            continue
        rows = flatten_rows(data)
        summaries.append(summarize_dataset(rows, ds))
        all_picked.extend(pick_cases(rows, dataset=ds))

    print("\n".join(summaries))
    print("\n---\n# 代表性案例\n")
    for tag, row in all_picked:
        print(fmt_row(tag, row))


if __name__ == "__main__":
    main()
