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
    numeric = _numeric_answer_match(pred, gold)
    if numeric is not None:
        return 1.0 if numeric else 0.0
    if g in p:
        return 1.0
    return 0.0


def _numeric_answer_match(pred: str, gold: str) -> bool | None:
    """Match short numeric answers by value instead of raw substring."""
    gold_duration = _duration_seconds(gold)
    if gold_duration is not None:
        pred_durations = _duration_seconds_all(pred)
        if pred_durations:
            return gold_duration in pred_durations

    gold_nums = re.findall(r"\d+(?:\.\d+)?", gold)
    if not gold_nums:
        return None
    if len(gold_nums) != 1:
        return normalize_answer(gold) in normalize_answer(pred)

    pred_nums = re.findall(r"\d+(?:\.\d+)?", pred)
    if not pred_nums:
        return False

    gold_num = float(gold_nums[0])
    gold_norm = normalize_answer(gold.replace(gold_nums[0], ""))
    pred_norm = normalize_answer(pred)
    if gold_norm in {"条", "次"}:
        return any(float(n) == gold_num for n in pred_nums) and (
            gold_norm in pred_norm or len(pred_nums) == 1
        )
    if gold_norm in {"元星", "元/星", "元"}:
        return any(abs(float(n) - gold_num) <= 0.05 for n in pred_nums) and (
            "元" in pred_norm or len(pred_nums) == 1
        )
    if gold_norm in {"秒", "分钟", "小时"}:
        return any(float(n) == gold_num for n in pred_nums) and (
            gold_norm in pred_norm or len(pred_nums) == 1
        )
    if not gold_norm:
        return any(float(n) == gold_num for n in pred_nums)
    # If the remaining gold text is not a numeric unit, the digit is part of
    # an entity name such as a book title. Fall back to normal span matching
    # instead of treating the answer as a pure numeric value.
    return None


def _duration_seconds(text: str) -> int | None:
    """Parse compact Chinese duration answers such as 5分00秒 or 480分钟."""
    values = _duration_seconds_all(text)
    return values[0] if values else None


def _duration_seconds_all(text: str) -> list[int]:
    """Parse all compact Chinese duration mentions in a string."""
    norm = normalize_answer(text)
    if not re.search(r"\d", norm):
        return []
    pattern = re.compile(r"(?:(\d+)小时)?(?:(\d+)分(?:钟)?)?(?:(\d+)秒)?")
    out: list[int] = []
    for m in pattern.finditer(norm):
        if not m.group(0) or not any(m.groups()):
            continue
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        out.append(hours * 3600 + minutes * 60 + seconds)
    return out


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
    """轻量 ROUGE-L (F1) — token 级 LCS，与学术标准 ROUGE-L 对齐。"""
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
    numeric_meta_keys = ("noise_ratio",)
    summaries: list[dict] = []
    for key, items in groups.items():
        summary = {g: k for g, k in zip(group_by, key)}
        summary["n"] = len(items)
        for mk in metric_keys:
            vals = [it[mk] for it in items if it.get(mk) is not None]
            summary[mk] = round(sum(vals) / len(vals), 4) if vals else None
        for mk in numeric_meta_keys:
            if mk in group_by:
                continue
            vals = [it[mk] for it in items if it.get(mk) is not None]
            if vals:
                summary[mk] = round(sum(vals) / len(vals), 4)
        summaries.append(summary)
    return summaries
