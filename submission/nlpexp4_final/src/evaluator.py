"""Answer evaluator —— DeepSeek LLM Judge（主指标）+ ISR/NAR + 可选语义归因。

问答生成使用 LM Studio 本地模型；Judge 固定走 DeepSeek，避免小模型自评失真。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from .config import CONFIG
from .llm_client import LLMClient, get_client, get_judge_client
from .metrics import attribute_answer, tokenize
from .metrics_semantic import attribute_answer_semantic
from .prompts import build_judge_prompt, is_long_form_dataset, normalize_dataset_key
from .rag_pipeline import RAGResult
from .utils import get_logger, parallel_map

logger = get_logger(__name__)


# ---------- 答案规范化（legacy） ----------

_PUNCT_PAT = re.compile(r"[\s，。、；：？！,\.;:?\!\"'`“”‘’（）()\[\]【】《》<>—\-]+")


def normalize_answer(text: str) -> str:
    """统一去标点、大小写、空白。"""
    if not text:
        return ""
    t = text.strip().lower()
    t = _PUNCT_PAT.sub("", t)
    return t


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) and normalize_answer(pred) == normalize_answer(gold) else 0.0


def _contains_match(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return 0.0
    return 1.0 if g in p else 0.0


def _token_f1(pred: str, gold: str) -> float:
    pt = tokenize(pred)
    gt = tokenize(gold)
    if not pt or not gt:
        return 0.0
    common: dict[str, int] = {}
    pt_count: dict[str, int] = {}
    gt_count: dict[str, int] = {}
    for t in pt:
        pt_count[t] = pt_count.get(t, 0) + 1
    for t in gt:
        gt_count[t] = gt_count.get(t, 0) + 1
    for t, c in pt_count.items():
        if t in gt_count:
            common[t] = min(c, gt_count[t])
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pt)
    r = n_common / len(gt)
    return 2 * p * r / (p + r)


def _rouge_l(pred: str, gold: str) -> float:
    a = tokenize(pred)
    b = tokenize(gold)
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def _best_over_golds(fn, pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(fn(pred, g) for g in golds)


def parse_judge_score(content: str) -> float | None:
    """从 judge 输出中解析 1-5 分，归一化到 [0, 1]。"""
    text = (content or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[1-5]", text):
        return float(text) / CONFIG.judge_score_max
    upper = text.upper()
    if upper in {"CORRECT", "YES", "TRUE"}:
        return 1.0
    if upper in {"INCORRECT", "WRONG", "NO", "FALSE"}:
        return 0.0
    m = re.search(r"(?<![0-9])([1-5])(?![0-9])", text)
    if m:
        return float(m.group(1)) / CONFIG.judge_score_max
    return None


def judge_correct_from_score(score: float | None, *, threshold: float = 0.8) -> float | None:
    """将 judge_score 转为二值正确率（默认 >=4/5 视为正确）。"""
    if score is None:
        return None
    return 1.0 if score >= threshold else 0.0


def cap_judge_score_for_abbreviated_pred(
    pred: str,
    golds: list[str],
    score: float | None,
    *,
    dataset: str | None = None,
    min_gold_len: int = 40,
    max_pred_ratio: float = 0.35,
    cap: float = 0.6,
) -> float | None:
    """结论被 pred 命中但过短省略 label 主体时，上限 3/5（防 Judge 过松）。

    长文数据集（BRIGHT/TEMPO 等）由专用 Judge prompt 约束，此处不再二次 cap。
    """
    if score is None:
        return None
    if is_long_form_dataset(dataset):
        return score
    p = (pred or "").strip()
    if not p or not golds:
        return score
    g = max((x for x in golds if x), key=len, default="")
    if len(g) < min_gold_len or len(p) >= len(g) * max_pred_ratio:
        return score
    if p in g or any(p in x for x in golds if x):
        return min(score, cap)
    return score


# ---------- 指标记录 ----------

@dataclass
class EvalMetrics:
    judge_score: float | None
    judge_correct: float | None
    isr: float
    nar: float
    isr_semantic: float | None = None
    nar_semantic: float | None = None
    n_semantic_units: int | None = None
    n_hallucination: int | None = None
    em: float | None = None
    contains: float | None = None
    token_f1: float | None = None
    rouge_l: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 主评估器 ----------

class Evaluator:
    """组合所有指标的统一入口。"""

    def __init__(
        self,
        *,
        use_llm_judge: bool = True,
        use_legacy_metrics: bool = False,
        use_semantic_attribution: bool = True,
        llm: LLMClient | None = None,
        judge_llm: LLMClient | None = None,
    ) -> None:
        self.use_llm_judge = use_llm_judge
        self.use_legacy_metrics = use_legacy_metrics
        self.use_semantic_attribution = use_semantic_attribution
        self.llm = llm or get_client()
        self.judge_llm = judge_llm or get_judge_client()

    def _llm_judge(
        self,
        query: str,
        pred: str,
        golds: list[str],
        *,
        language: str,
        dataset: str | None = None,
        subset: str | None = None,
    ) -> float | None:
        if not self.use_llm_judge:
            return None
        system, user = build_judge_prompt(
            query, pred, golds, language=language, dataset=dataset, subset=subset
        )
        try:
            out = self.judge_llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model=CONFIG.judge_model,
                max_tokens=16,
            )
            return parse_judge_score(out.get("content") or "")
        except Exception as e:
            logger.warning(f"LLM-judge failed: {e}")
        return None

    def evaluate_one(self, result: RAGResult, *, language: str = "zh") -> EvalMetrics:
        pred = result.prediction
        golds = result.gold_answers
        dataset = normalize_dataset_key(result.metadata.get("dataset"))
        subset = result.metadata.get("subset")
        if not (pred or "").strip():
            judge = 0.0
            judge_correct = 0.0
        else:
            judge = self._llm_judge(
                result.query,
                pred,
                golds,
                language=language,
                dataset=dataset,
                subset=subset,
            )
            judge = cap_judge_score_for_abbreviated_pred(
                pred, golds, judge, dataset=dataset
            )
            judge_correct = judge_correct_from_score(judge)
        attr = attribute_answer(pred, result.docs, result.labels)

        isr_sem = nar_sem = None
        n_units = n_hallu = None
        if self.use_semantic_attribution and pred.strip() and result.docs:
            try:
                sem = attribute_answer_semantic(
                    result.query, pred, result.docs, result.labels, llm=self.judge_llm
                )
                isr_sem = sem.isr_semantic
                nar_sem = sem.nar_semantic
                n_units = sem.n_units
                n_hallu = sem.n_hallucination
            except Exception as e:
                logger.warning(f"semantic attribution failed sample {result.sample_id}: {e}")

        em = cont = f1 = rl = None
        if self.use_legacy_metrics:
            em = _best_over_golds(_exact_match, pred, golds)
            cont = _best_over_golds(_contains_match, pred, golds)
            f1 = _best_over_golds(_token_f1, pred, golds)
            rl = _best_over_golds(_rouge_l, pred, golds)

        return EvalMetrics(
            judge_score=judge,
            judge_correct=judge_correct,
            isr=attr.isr,
            nar=attr.nar,
            isr_semantic=isr_sem,
            nar_semantic=nar_sem,
            n_semantic_units=n_units,
            n_hallucination=n_hallu,
            em=em,
            contains=cont,
            token_f1=f1,
            rouge_l=rl,
        )

    def evaluate_batch(
        self,
        results: Iterable[RAGResult],
        *,
        language: str = "zh",
        workers: int = 1,
        show_progress: bool = True,
    ) -> list[dict]:
        items = list(results)

        def _one(r: RAGResult) -> dict:
            m = self.evaluate_one(r, language=language)
            return {
                "sample_id": r.sample_id,
                "query": r.query,
                "noise_ratio": r.noise_ratio,
                "noise_type": r.noise_type,
                "noise_position": r.noise_position,
                "method": r.metadata.get("method", "naive"),
                "prediction": r.prediction,
                "gold": r.gold_answers,
                "docs": r.docs,
                "labels": r.labels,
                **m.to_dict(),
            }

        if not items:
            return []
        if len(items) == 1 or workers <= 1:
            if show_progress and len(items) > 1:
                from tqdm import tqdm

                return [_one(r) for r in tqdm(items, desc="Evaluate/judge")]
            return [_one(r) for r in items]
        return parallel_map(items, _one, workers=workers, show_progress=show_progress, desc="Evaluate/judge")


# ---------- 汇总 ----------

PRIMARY_SCORE_KEY = "judge_score"

_METRIC_KEYS = (
    "judge_score",
    "judge_correct",
    "em",
    "contains",
    "token_f1",
    "rouge_l",
    "isr",
    "nar",
    "isr_semantic",
    "nar_semantic",
    "n_hallucination",
)


def aggregate(rows: list[dict], *, group_by: tuple[str, ...] = ("method", "noise_ratio")) -> list[dict]:
    """按指定字段分组求平均（去 None）。"""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(g) for g in group_by)
        groups.setdefault(key, []).append(row)

    summaries: list[dict] = []
    for key, items in groups.items():
        summary = {g: k for g, k in zip(group_by, key)}
        summary["n"] = len(items)
        for mk in _METRIC_KEYS:
            vals = [it[mk] for it in items if it.get(mk) is not None]
            summary[mk] = round(sum(vals) / len(vals), 4) if vals else None
        summaries.append(summary)
    return summaries
