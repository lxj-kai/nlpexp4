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
    noise_resistance_slope,
    noise_sensitivity,
    correction_recovery_rate,
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
            summary = aggregate(rows, group_by=("method",))[0] if rows else {}
        out.append(RunResult(condition=cond, rows=rows, elapsed=t.elapsed, summary=summary))
    return out


def compute_robustness_table(results: list[RunResult]) -> list[dict]:
    """从多 condition 结果中提取 NS / NRS / CRR 等 method-level 指标。"""
    methods = sorted({r.condition.method for r in results})
    out: list[dict] = []
    for method in methods:
        ms = [r for r in results if r.condition.method == method]
        ratio_score = [
            (r.condition.noise_ratio, r.summary.get("token_f1", 0.0) or 0.0) for r in ms
        ]
        clean = next((s for r, s in ratio_score if r == 0.0), None)
        noisy_avg = (
            sum(s for r, s in ratio_score if r > 0) / max(1, sum(1 for r, _ in ratio_score if r > 0))
            if any(r > 0 for r, _ in ratio_score)
            else None
        )
        nrs = noise_resistance_slope(
            [r for r, _ in ratio_score], [s for _, s in ratio_score]
        )
        ns = (
            noise_sensitivity(clean, noisy_avg)
            if clean is not None and noisy_avg is not None
            else None
        )
        isr_avg = sum(r.summary.get("isr", 0.0) or 0.0 for r in ms) / len(ms)
        nar_avg = sum(r.summary.get("nar", 0.0) or 0.0 for r in ms) / len(ms)
        out.append(
            {
                "method": method,
                "score_clean": clean,
                "score_noisy_avg": noisy_avg,
                "NS": ns,
                "NRS": nrs,
                "ISR_avg": round(isr_avg, 4),
                "NAR_avg": round(nar_avg, 4),
                "n_conditions": len(ms),
            }
        )
    return out


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
