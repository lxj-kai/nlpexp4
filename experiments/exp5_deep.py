"""深度实验集成测试：验证四个深化方向。

方向1：自适应矫正器 (adaptive)
方向2：语义级ISR/NAR
方向3：Confidence消融对比 (4个变体)
方向4：迭代自纠正循环 (iterative_sc)

用法：python -m experiments.exp5_deep --n 30 --language zh
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from experiments._runner import (
    RunCondition,
    load_corpus,
    run_conditions,
    save_run,
)
from src.config import CONFIG
from src.utils import get_logger
from src.metrics_semantic import attribute_answer_semantic, compare_attributions
from src.metrics import attribute_answer

logger = get_logger("exp5")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--ratio", type=float, default=0.75)
    args = p.parse_args()

    records = load_corpus(language=args.language, subset="main", limit=args.n)

    # ============================================
    # Phase 1: 自适应矫正器测试 (semantic + CF)
    # ============================================
    logger.info("=" * 50)
    logger.info("Phase 1/4: Adaptive Corrector")
    logger.info("=" * 50)

    conditions_adaptive = [
        RunCondition(method="naive", noise_ratio=args.ratio, noise_type="semantic", label="naive_sem"),
        RunCondition(method="naive", noise_ratio=args.ratio, noise_type="counterfactual", label="naive_cf"),
        RunCondition(method="confidence", noise_ratio=args.ratio, noise_type="semantic", label="confidence_sem"),
        RunCondition(method="adaptive", noise_ratio=args.ratio, noise_type="semantic", label="adaptive_sem"),
        RunCondition(method="adaptive", noise_ratio=args.ratio, noise_type="counterfactual", label="adaptive_cf"),
        RunCondition(method="prompt", noise_ratio=args.ratio, noise_type="counterfactual", label="prompt_cf"),
    ]

    results_adaptive = run_conditions(
        records=records, conditions=conditions_adaptive,
        language=args.language, show_progress=True,
    )

    # 统计路由决策
    route_counts = defaultdict(int)
    for r in results_adaptive:
        if r.condition.method != "adaptive":
            continue
        for row in r.rows:
            route = row.get("adaptive_route", "UNKNOWN")
            route_counts[route] += 1
    logger.info(f"Adaptive routing: {dict(route_counts)}")

    path1 = save_run(
        experiment_name=f"exp5_deep_adaptive_{args.language}",
        results=results_adaptive,
        extras={"phase": "adaptive", "route_distribution": dict(route_counts)},
    )
    print(f"saved adaptive → {path1}")

    # ============================================
    # Phase 2: 语义级ISR/NAR (对confidence结果做深度溯源)
    # ============================================
    logger.info("=" * 50)
    logger.info("Phase 2/4: Semantic Attribution")
    logger.info("=" * 50)

    # 从 Phase 1 的 confidence_sem 结果中取前5条做语义溯源
    conf_result = next(r for r in results_adaptive if r.condition.label == "confidence_sem")
    attributions = []
    for row in conf_result.rows[:5]:
        pred = row.get("prediction", "")
        query = row.get("query", "")
        if not pred or not query:
            continue

        # 字符级溯源
        from src.data_loader import load_dataset
        from src.noise_injector import inject

        rec = next(r for r in load_dataset(language=args.language, subset="main", limit=args.n + 10) if r.id == row["sample_id"])
        ctx = inject(rec, noise_ratio=args.ratio, noise_type="semantic", max_docs=CONFIG.max_docs)

        token_attr = attribute_answer(pred, ctx.docs, list(ctx.labels))
        sem_attr = attribute_answer_semantic(query, pred, ctx.docs, list(ctx.labels))

        comparison = compare_attributions(token_attr, sem_attr)
        comparison["sample_id"] = row["sample_id"]
        comparison["query"] = query[:60]
        comparison["prediction"] = pred
        comparison["gold"] = row.get("gold", "")
        attributions.append(comparison)

        logger.info(f"  sample {row['sample_id']}: token ISR={token_attr.isr:.3f} → semantic ISR={sem_attr.isr_semantic:.3f} "
                     f"| NAR {token_attr.nar:.3f} → {sem_attr.nar_semantic:.3f} "
                     f"| hallucinations={sem_attr.n_hallucination}")

    path2 = save_run(
        experiment_name=f"exp5_deep_attribution_{args.language}",
        results=results_adaptive[:2],  # placeholder
        extras={"phase": "attribution", "attributions": attributions},
    )
    print(f"saved attribution → {path2}")

    # ============================================
    # Phase 3: 消融实验 (semantic r=0.75, n=20)
    # ============================================
    logger.info("=" * 50)
    logger.info("Phase 3/4: Ablation Study")
    logger.info("=" * 50)

    ablation_records = load_corpus(language=args.language, subset="main", limit=min(args.n, 20))
    conditions_ablation = [
        RunCondition(method="naive", noise_ratio=args.ratio, noise_type="semantic", label="naive"),
        RunCondition(method="confidence", noise_ratio=args.ratio, noise_type="semantic", label="confidence"),
        RunCondition(method="ablated_full", noise_ratio=args.ratio, noise_type="semantic", label="ablated_full"),
        RunCondition(method="ablated_no_decompose", noise_ratio=args.ratio, noise_type="semantic", label="ablated_no_dec"),
        RunCondition(method="ablated_no_evidence", noise_ratio=args.ratio, noise_type="semantic", label="ablated_no_ev"),
        RunCondition(method="ablated_no_tag", noise_ratio=args.ratio, noise_type="semantic", label="ablated_no_tag"),
    ]

    results_ablation = run_conditions(
        records=ablation_records, conditions=conditions_ablation,
        language=args.language, show_progress=True,
    )

    path3 = save_run(
        experiment_name=f"exp5_deep_ablation_{args.language}",
        results=results_ablation,
        extras={"phase": "ablation"},
    )
    print(f"saved ablation → {path3}")

    # ============================================
    # Phase 4: 迭代自纠正 (semantic r=0.75, n=10)
    # ============================================
    logger.info("=" * 50)
    logger.info("Phase 4/4: Iterative Self-Correction")
    logger.info("=" * 50)

    sc_records = load_corpus(language=args.language, subset="main", limit=min(args.n, 10))
    conditions_sc = [
        RunCondition(method="naive", noise_ratio=args.ratio, noise_type="semantic", label="naive"),
        RunCondition(method="iterative_sc", noise_ratio=args.ratio, noise_type="semantic", label="iterative_sc"),
        RunCondition(method="iterative_sc", noise_ratio=args.ratio, noise_type="counterfactual", label="iterative_sc_cf"),
    ]

    results_sc = run_conditions(
        records=sc_records, conditions=conditions_sc,
        language=args.language, show_progress=True,
    )

    # 统计收敛情况
    sc_stats = {"converged": 0, "total_rounds": []}
    for r in results_sc:
        if r.condition.method != "iterative_sc":
            continue
        for row in r.rows:
            metadata = row.get("metadata", {})
            if metadata.get("converged"):
                sc_stats["converged"] += 1
            sc_stats["total_rounds"].append(metadata.get("rounds", 0))
    logger.info(f"Self-correct stats: converged={sc_stats['converged']}/{len(sc_stats['total_rounds'])}, "
                 f"avg rounds={sum(sc_stats['total_rounds'])/max(1,len(sc_stats['total_rounds'])):.1f}")

    path4 = save_run(
        experiment_name=f"exp5_deep_selfcorrect_{args.language}",
        results=results_sc,
        extras={"phase": "self_correct", "convergence": sc_stats},
    )
    print(f"saved self-correct → {path4}")

    print("\n" + "=" * 50)
    print("EXP5 COMPLETE — all 4 deep directions tested")
    print("=" * 50)


if __name__ == "__main__":
    main()
