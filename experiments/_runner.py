"""通用实验 runner —— 把"加载数据 → 注入噪音 → 运行方法 → 评估 → 落盘"流程沉淀下来。

实验脚本只负责声明"我要跑哪些配置"，具体执行交给本模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from src.config import CONFIG
from src.correctors import get_corrector
from src.data_loader import Language, Subset, load_dataset, RGBRecord
from src.evaluator import Evaluator, aggregate
from src.llm_client import LLMClient
from src.metrics import (
    correction_recovery_rate,
    noise_resistance_slope,
    noise_sensitivity,
)
from src.noise_injector import (
    NoisePosition,
    NoiseType,
    batch_inject,
)
from src.rag_pipeline import RAGPipeline, RAGResult
from src.utils import Timer, get_logger, now_tag, set_seed, write_json

logger = get_logger(__name__)


@dataclass
class RunCondition:
    """一组实验条件（一行表格里的一格）。"""

    method: str = "naive"
    noise_ratio: float = 0.0
    noise_type: NoiseType = "semantic"
    noise_position: NoisePosition = "interleave"
    label: str = ""

    def key(self) -> tuple:
        return (self.method, self.noise_ratio, self.noise_type, self.noise_position)

    def short(self) -> str:
        return (
            self.label
            or f"{self.method}|r={self.noise_ratio}|t={self.noise_type}|p={self.noise_position}"
        )


@dataclass
class RunResult:
    """单次 condition 的运行结果。"""

    condition: RunCondition
    rows: list[dict]
    elapsed: float
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition": asdict(self.condition),
            "summary": self.summary,
            "elapsed": round(self.elapsed, 2),
            "n": len(self.rows),
            "rows": self.rows,
        }


def _run_method(
    method: str,
    contexts,
    *,
    llm: LLMClient,
    language: str,
    show_progress: bool,
) -> list[RAGResult]:
    if method == "naive":
        pipe = RAGPipeline(llm=llm)
        return pipe.batch_answer(contexts, language=language, show_progress=show_progress)
    corr = get_corrector(method, llm=llm)
    return corr.batch_correct(contexts, language=language, show_progress=show_progress)


def run_conditions(
    *,
    records: Sequence[RGBRecord],
    conditions: Sequence[RunCondition],
    llm: LLMClient | None = None,
    evaluator: Evaluator | None = None,
    language: str = "zh",
    show_progress: bool = True,
) -> list[RunResult]:
    """主循环：对每个 condition 跑全部 records。"""
    llm = llm or LLMClient()
    evaluator = evaluator or Evaluator(use_llm_judge=False, llm=llm)
    out: list[RunResult] = []
    for cond in conditions:
        with Timer(cond.short()) as t:
            ctxs = batch_inject(
                list(records),
                noise_ratio=cond.noise_ratio,
                noise_type=cond.noise_type,
                noise_position=cond.noise_position,
            )
            results = _run_method(
                cond.method, ctxs, llm=llm, language=language, show_progress=show_progress
            )
            rows = evaluator.evaluate_batch(results)
            for r in rows:
                r["method"] = cond.method
                r["noise_ratio_target"] = cond.noise_ratio
                r["noise_type"] = cond.noise_type
                r["noise_position"] = cond.noise_position
            summary = (
                aggregate(
                    rows,
                    group_by=("method", "noise_ratio_target", "noise_type", "noise_position"),
                )[0]
                if rows
                else {}
            )
        out.append(RunResult(condition=cond, rows=rows, elapsed=t.elapsed, summary=summary))
    return out


def compute_robustness_table(
    results: list[RunResult],
    *,
    score_key: str = "token_f1",
) -> list[dict]:
    """从多 condition 结果中提取 NS / NRS / CRR / ISR / NAR 鲁棒性指标。

    切片维度：(method, noise_type)。同一 method 下不同 noise_type 的
    噪音梯度曲线斜率独立计算，避免把 semantic / counterfactual / mixed
    搅在一起做线性回归。

    CRR 需要同 method+noise_type 下既有 clean (ratio=0) 又有 noisy 的结果，
    且会尝试从同 noise_type 下寻找 corrected method 与 naive baseline 的配对。
    """

    def cond_key(r: RunResult) -> tuple[str, str]:
        return (r.condition.method, r.condition.noise_type)

    keys = sorted({cond_key(r) for r in results})

    naive_clean_scores: dict[str, float] = {}
    naive_noisy_scores: dict[str, float] = {}
    for method, ntype in keys:
        if method != "naive":
            continue
        ms = [r for r in results if cond_key(r) == (method, ntype)]
        for r in ms:
            score = float(r.summary.get(score_key, 0.0) or 0.0)
            if r.condition.noise_ratio == 0.0:
                naive_clean_scores[ntype] = score
            elif r.condition.noise_ratio > 0:
                naive_noisy_scores.setdefault(ntype, [])
                naive_noisy_scores[ntype].append(score)  # type: ignore[union-attr]

    naive_noisy_avg: dict[str, float] = {}
    for ntype, scores_list in naive_noisy_scores.items():
        if isinstance(scores_list, list) and scores_list:
            naive_noisy_avg[ntype] = sum(scores_list) / len(scores_list)

    out: list[dict] = []
    for method, ntype in keys:
        ms = [r for r in results if cond_key(r) == (method, ntype)]
        ratio_score = sorted(
            (r.condition.noise_ratio, float(r.summary.get(score_key, 0.0) or 0.0))
            for r in ms
        )
        ratios = [r for r, _ in ratio_score]
        scores = [s for _, s in ratio_score]

        clean = next((s for r, s in ratio_score if r == 0.0), None)
        positive_ratio_scores = [s for r, s in ratio_score if r > 0]
        noisy_avg = (
            sum(positive_ratio_scores) / len(positive_ratio_scores)
            if positive_ratio_scores
            else None
        )

        nrs = (
            noise_resistance_slope(ratios, scores)
            if len({r for r in ratios}) >= 2
            else None
        )
        ns = (
            noise_sensitivity(clean, noisy_avg)
            if clean is not None and noisy_avg is not None
            else None
        )

        crr = None
        if method != "naive" and noisy_avg is not None:
            s_clean = naive_clean_scores.get(ntype)
            s_noisy = naive_noisy_avg.get(ntype)
            if s_clean is not None and s_noisy is not None:
                crr = correction_recovery_rate(s_clean, s_noisy, noisy_avg)

        isr_avg = (
            sum(r.summary.get("isr", 0.0) or 0.0 for r in ms) / len(ms) if ms else 0.0
        )
        nar_avg = (
            sum(r.summary.get("nar", 0.0) or 0.0 for r in ms) / len(ms) if ms else 0.0
        )

        noise_ratio_actual = [
            r.summary.get("noise_ratio", r.condition.noise_ratio)
            for r in ms if r.condition.noise_ratio > 0
        ]
        noise_ratio_stats = None
        if noise_ratio_actual:
            import numpy as _np
            arr = _np.array([float(v) for v in noise_ratio_actual if v is not None])
            if len(arr):
                noise_ratio_stats = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                }

        out.append(
            {
                "method": method,
                "noise_type": ntype,
                "score_metric": score_key,
                "score_clean": clean,
                "score_noisy_avg": noisy_avg,
                "NS": None if ns is None else round(ns, 4),
                "NRS": None if nrs is None else round(nrs, 4),
                "CRR": None if crr is None else round(crr, 4),
                "ISR_avg": round(isr_avg, 4),
                "NAR_avg": round(nar_avg, 4),
                "n_conditions": len(ms),
                "ratios": ratios,
                "scores": [round(s, 4) for s in scores],
                "noise_ratio_actual": noise_ratio_stats,
            }
        )
    return out


def _git_hash() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(CONFIG.project_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def save_run(
    *,
    experiment_name: str,
    results: list[RunResult],
    extras: dict[str, Any] | None = None,
) -> str:
    CONFIG.ensure_dirs()
    out = {
        "experiment": experiment_name,
        "timestamp": now_tag(),
        "git_commit": _git_hash(),
        "config": CONFIG.to_dict(),
        "results": [r.to_dict() for r in results],
        "robustness_table": compute_robustness_table(results),
    }
    if extras:
        out.update(extras)
    fname = f"{experiment_name}_{out['timestamp']}.json"
    path = CONFIG.results_dir / fname
    write_json(out, path)
    logger.info(f"results -> {path}")
    return str(path)


def load_corpus(
    language: Language = "zh",
    subset: Subset = "main",
    *,
    limit: int | None = None,
) -> list[RGBRecord]:
    set_seed(CONFIG.seed)
    return load_dataset(language=language, subset=subset, limit=limit)
