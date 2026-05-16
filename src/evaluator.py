"""Answer evaluator —— EM / Token-F1 / ROUGE-L / LLM-as-Judge + 汇总。

针对 RGB 数据集 answer 可能为 list[str]（多答案 OR 关系），EM 与 F1 取最大值。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from .config import CONFIG
from .llm_client import LLMClient, get_client
from .metrics import attribute_answer, tokenize
from .prompts import JUDGE_SYSTEM_ZH, JUDGE_USER_TMPL
from .rag_pipeline import RAGResult
from .utils import get_logger

logger = get_logger(__name__)


# ---------- 答案规范化 ----------

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
    """更宽容的"包含匹配"：模型输出多了几个字也算对（适合短答案场景）。"""
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
    """轻量 ROUGE-L (F1) — 字符级 LCS，避免引入 rouge 库的依赖问题。"""
    a = list(normalize_answer(pred))
    b = list(normalize_answer(gold))
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


# ---------- 指标记录 ----------

@dataclass
class EvalMetrics:
    em: float
    contains: float
    token_f1: float
    rouge_l: float
    judge_score: float | None
    isr: float
    nar: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 主评估器 ----------

class Evaluator:
    """组合所有指标的统一入口。"""

    def __init__(self, *, use_llm_judge: bool = False, llm: LLMClient | None = None) -> None:
        self.use_llm_judge = use_llm_judge
        self.llm = llm if llm is not None else (get_client() if use_llm_judge else None)

    def _llm_judge(self, query: str, pred: str, golds: list[str]) -> float | None:
        if not self.use_llm_judge or self.llm is None:
            return None
        gold_str = " / ".join(golds) if golds else ""
        try:
            out = self.llm.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_ZH},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TMPL.format(query=query, gold=gold_str, pred=pred),
                    },
                ],
                model=CONFIG.judge_model,
                max_tokens=8,
            )
            content = (out["content"] or "").strip()
            m = re.search(r"[1-5]", content)
            if m:
                return float(m.group(0)) / CONFIG.judge_score_max
        except Exception as e:
            logger.warning(f"LLM-judge failed: {e}")
        return None

    def evaluate_one(self, result: RAGResult) -> EvalMetrics:
        pred = result.prediction
        golds = result.gold_answers
        em = _best_over_golds(_exact_match, pred, golds)
        cont = _best_over_golds(_contains_match, pred, golds)
        f1 = _best_over_golds(_token_f1, pred, golds)
        rl = _best_over_golds(_rouge_l, pred, golds)
        attr = attribute_answer(pred, result.docs, result.labels)
        judge = self._llm_judge(result.query, pred, golds)
        return EvalMetrics(em, cont, f1, rl, judge, attr.isr, attr.nar)

    def evaluate_batch(self, results: Iterable[RAGResult]) -> list[dict]:
        rows: list[dict] = []
        for r in results:
            m = self.evaluate_one(r)
            rows.append(
                {
                    "sample_id": r.sample_id,
                    "noise_ratio": r.noise_ratio,
                    "noise_type": r.noise_type,
                    "noise_position": r.noise_position,
                    "method": r.metadata.get("method", "naive"),
                    "prediction": r.prediction,
                    "gold": r.gold_answers,
                    **m.to_dict(),
                }
            )
        return rows


# ---------- 汇总 ----------

def aggregate(rows: list[dict], *, group_by: tuple[str, ...] = ("method", "noise_ratio")) -> list[dict]:
    """按指定字段分组求平均（去 None）。"""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(g) for g in group_by)
        groups.setdefault(key, []).append(row)

    metric_keys = ("em", "contains", "token_f1", "rouge_l", "judge_score", "isr", "nar")
    summaries: list[dict] = []
    for key, items in groups.items():
        summary = {g: k for g, k in zip(group_by, key)}
        summary["n"] = len(items)
        for mk in metric_keys:
            vals = [it[mk] for it in items if it.get(mk) is not None]
            summary[mk] = round(sum(vals) / len(vals), 4) if vals else None
        summaries.append(summary)
    return summaries
