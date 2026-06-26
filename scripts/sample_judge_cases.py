"""跑少量样本并打印 Judge 案例（生成 LM Studio + 审查 DeepSeek）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services import verdict
from src.data_loader import load_dataset
from src.evaluator import Evaluator
from src.llm_client import LLMClient, get_judge_client
from src.noise_injector import inject
from src.rag_pipeline import RAGPipeline

CASES = [
    # (dataset, lang, subset, sample_id, noise_ratio, note)
    ("rgb", "zh", "main", None, 0.75, "RGB 高噪音"),
    ("rgb", "zh", "main", None, 0.0, "RGB 无噪音"),
    ("cmedqa", "zh", "main", None, 0.5, "CMedQA 中噪音"),
    ("cmedqa", "zh", "main", None, 0.75, "CMedQA 高噪音"),
    ("2wiki", "en", "main", None, 0.5, "2Wiki 中噪音"),
]


def _verdict_label(metrics) -> str:
    v = verdict(metrics)
    return {"correct": "✅ 正确", "partial": "⚠️ 部分", "wrong": "❌ 错误", "noise_biased": "❌ 噪音主导"}.get(v, v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-ids", default="", help="逗号分隔 sample id，优先跑这些")
    p.add_argument("--per-dataset", type=int, default=2, help="每个数据集额外随机样本数")
    args = p.parse_args()

    llm = LLMClient()
    judge = get_judge_client()
    ev = Evaluator(llm=llm, judge_llm=judge, use_semantic_attribution=False)
    rag = RAGPipeline(llm=llm)

    rows: list[dict] = []

    # 固定抽前 N 条 + 指定 id
    specs = [
        ("rgb", "zh", "main", [0, 50, 120]),
        ("cmedqa", "zh", "main", [10, 200]),
        ("2wiki", "en", "main", [5, 100]),
    ]
    if args.seed_ids:
        for sid in args.seed_ids.split(","):
            specs.append(("rgb", "zh", "main", [int(sid.strip())]))

    seen: set[tuple] = set()
    for dataset, lang, subset, ids in specs:
        records = {r.id: r for r in load_dataset(language=lang, subset=subset, dataset=dataset, limit=500)}
        for sid in ids:
            if sid not in records:
                continue
            for ratio in (0.0, 0.75):
                key = (dataset, sid, ratio)
                if key in seen:
                    continue
                seen.add(key)
                rec = records[sid]
                ctx = inject(rec, noise_ratio=ratio, noise_type="semantic", noise_position="interleave")
                result = rag.answer(ctx, language=lang)
                m = ev.evaluate_one(result, language=lang)
                gold = " / ".join(rec.answers_norm[:2])
                if len(rec.answers_norm) > 2:
                    gold += " …"
                rows.append(
                    {
                        "dataset": dataset,
                        "id": sid,
                        "ratio": ratio,
                        "query": rec.query[:80] + ("…" if len(rec.query) > 80 else ""),
                        "gold": gold[:120],
                        "pred": result.prediction[:120],
                        "judge_score": m.judge_score,
                        "judge_correct": m.judge_correct,
                        "isr": m.isr,
                        "nar": m.nar,
                        "verdict": _verdict_label(m),
                    }
                )

    print(f"\n共 {len(rows)} 条案例（DeepSeek Judge）\n")
    print("=" * 100)
    for i, r in enumerate(rows, 1):
        print(f"\n【案例 {i}】{r['dataset']} #{r['id']} · noise={r['ratio']}")
        print(f"问题：{r['query']}")
        print(f"参考答案：{r['gold']}")
        print(f"模型预测：{r['pred']}")
        js = f"{r['judge_score']:.2f}" if r['judge_score'] is not None else "—"
        jc = "是" if r["judge_correct"] == 1 else ("否" if r["judge_correct"] == 0 else "—")
        print(f"Judge：score={js} · 正确={jc}")
        print(f"ISR={r['isr']:.3f} · NAR={r['nar']:.3f} · 判定 {r['verdict']}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
