"""Robustness metrics — 5 个自主提出的鲁棒性指标。

公式（与 PLAN.html 一致）：
- NS  = (S_clean - S_noisy) / S_clean          噪音敏感度
- NRS = ΔScore / ΔNoiseRatio                  噪音抵抗曲线斜率
- ISR = positive 文档贡献 token / 答案 token   信息溯源率
- NAR = negative 文档贡献 token / 答案 token   噪音采纳率
- CRR = (S_corr - S_noisy) / (S_clean - S_noisy)  矫正恢复率
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_CHINESE_RANGE = ("\u4e00", "\u9fff")
_TOKEN_PAT = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """语言无关的轻量 token 化：英文按词，中文按字。"""
    if not text:
        return []
    return _TOKEN_PAT.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


# ---------- NS / NRS ----------

def noise_sensitivity(score_clean: float, score_noisy: float) -> float:
    """NS：噪音引起的相对性能下降幅度。范围 (-∞, 1]，越大越敏感。"""
    return _safe_div(score_clean - score_noisy, score_clean, 0.0)


def noise_resistance_slope(ratios: list[float], scores: list[float]) -> float:
    """NRS：score 对 noise_ratio 的线性回归斜率。斜率越接近 0 越鲁棒。"""
    if len(ratios) < 2:
        return 0.0
    x = np.asarray(ratios, dtype=float)
    y = np.asarray(scores, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


# ---------- ISR / NAR (token 级溯源) ----------

@dataclass
class SourceAttribution:
    """答案信息的来源归因结果。"""

    n_answer_tokens: int
    n_from_positive: int
    n_from_negative: int
    n_from_neither: int
    isr: float
    nar: float


def attribute_answer(
    answer: str,
    docs: list[str],
    labels: list[str],
    *,
    min_token_len: int = 2,
) -> SourceAttribution:
    """把答案 token 归因到 positive / negative / 无 三类来源。

    简化做法：对答案中的每个 token，看它是否出现在某类文档里；同时命中两类
    时按 positive 优先（更保守的 ISR 估计）。
    """
    ans_tokens = [t for t in tokenize(answer) if len(t) >= min_token_len]
    if not ans_tokens:
        return SourceAttribution(0, 0, 0, 0, 0.0, 0.0)

    pos_text = " ".join(d for d, l in zip(docs, labels) if l == "positive")
    neg_text = " ".join(d for d, l in zip(docs, labels) if l != "positive")
    pos_set = token_set(pos_text)
    neg_set = token_set(neg_text)

    n_pos = n_neg = n_neither = 0
    for tk in ans_tokens:
        in_pos = tk in pos_set
        in_neg = tk in neg_set
        if in_pos:
            n_pos += 1
        elif in_neg:
            n_neg += 1
        else:
            n_neither += 1

    total = len(ans_tokens)
    return SourceAttribution(
        n_answer_tokens=total,
        n_from_positive=n_pos,
        n_from_negative=n_neg,
        n_from_neither=n_neither,
        isr=_safe_div(n_pos, total, 0.0),
        nar=_safe_div(n_neg, total, 0.0),
    )


# ---------- CRR ----------

def correction_recovery_rate(
    score_clean: float, score_noisy: float, score_corrected: float
) -> float:
    """CRR：矫正机制相对"无噪音上限"恢复了多少损失。

    特例：
    - 无下降（clean==noisy）：返回 0（无可恢复）
    - 矫正后超过 clean：返回 >1（罕见的“正向噪音”现象）
    """
    denom = score_clean - score_noisy
    if abs(denom) < 1e-9:
        return 0.0
    return (score_corrected - score_noisy) / denom


# ---------- 汇总 ----------

@dataclass
class RobustnessSummary:
    """同一矫正方法在某数据集上的鲁棒性汇总。"""

    method: str
    score_clean: float
    score_noisy_avg: float
    ns: float
    nrs: float
    isr_avg: float
    nar_avg: float
    crr_avg: float | None = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "score_clean": round(self.score_clean, 4),
            "score_noisy_avg": round(self.score_noisy_avg, 4),
            "NS": round(self.ns, 4),
            "NRS": round(self.nrs, 4),
            "ISR_avg": round(self.isr_avg, 4),
            "NAR_avg": round(self.nar_avg, 4),
            "CRR_avg": None if self.crr_avg is None else round(self.crr_avg, 4),
            "n_samples": self.n_samples,
        }


def aggregate_robustness(
    *,
    method: str,
    score_clean: float,
    ratio_score_pairs: list[tuple[float, float]],
    isr_values: list[float],
    nar_values: list[float],
    score_corrected: float | None = None,
    score_noisy_for_crr: float | None = None,
    n_samples: int = 0,
) -> RobustnessSummary:
    ratios = [r for r, _ in ratio_score_pairs]
    scores = [s for _, s in ratio_score_pairs]
    noisy_avg = float(np.mean([s for r, s in ratio_score_pairs if r > 0])) if any(r > 0 for r in ratios) else score_clean
    ns = noise_sensitivity(score_clean, noisy_avg)
    nrs = noise_resistance_slope(ratios, scores)
    crr = None
    if score_corrected is not None and score_noisy_for_crr is not None:
        crr = correction_recovery_rate(score_clean, score_noisy_for_crr, score_corrected)
    return RobustnessSummary(
        method=method,
        score_clean=score_clean,
        score_noisy_avg=noisy_avg,
        ns=ns,
        nrs=nrs,
        isr_avg=float(np.mean(isr_values)) if isr_values else 0.0,
        nar_avg=float(np.mean(nar_values)) if nar_values else 0.0,
        crr_avg=crr,
        n_samples=n_samples,
    )
