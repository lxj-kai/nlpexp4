"""Robustness metrics — 5 个自主提出的鲁棒性指标。

公式（与 PLAN.html 一致）：
- NS  = (S_clean - S_noisy) / S_clean          噪音敏感度
- NRS = ΔScore / ΔNoiseRatio                  噪音抵抗曲线斜率
- ISR = positive 文档贡献片段 / 答案有效片段   信息溯源率
- NAR = negative 文档贡献片段 / 答案有效片段   噪音采纳率
- CRR = (S_corr - S_noisy) / (S_clean - S_noisy)  矫正恢复率

ISR/NAR 归因策略（v2）：
- 中文以 bigram (2-gram) 为最小溯源单位，避免单字"的/是/在"全文必中导致 ISR 虚高
- 英文/数字按词
- 过滤高频功能字 / stopword
- 去重：每个 unique 片段最多计一次
- 命中两类：按片段在 positive vs negative 的出现次数差额分配，不再无脑 positive 优先
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

_CHINESE_RANGE = ("\u4e00", "\u9fff")
_TOKEN_PAT = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_CJK_PAT = re.compile(r"[\u4e00-\u9fff]")

CN_STOPCHARS: frozenset[str] = frozenset(
    "的了是在和与或及对于把被这那些个其之也都就还又或而但即如则若以及"
    "我你他她它我们你们他们她们它们自己也是因为所以但是然后于是因此"
    "一二三四五六七八九十百千万年月日时分秒上下左右前后内外中之间"
    "可能应该或许大概一种一些什么怎么如何为何为什么"
)
EN_STOPWORDS: frozenset[str] = frozenset(
    "a an the of to in on at by for from with and or but if then so as is "
    "are was were be been being have has had do does did this that these "
    "those it its their there here which who whom whose what when where why how"
    .split()
)


def tokenize(text: str) -> list[str]:
    """语言无关的轻量 token 化：英文按词，中文按字。"""
    if not text:
        return []
    return _TOKEN_PAT.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def _is_cjk(token: str) -> bool:
    return bool(token) and _CJK_PAT.fullmatch(token) is not None


def attribution_grams(text: str) -> list[str]:
    """提取用于 ISR/NAR 归因的"有意义片段"。

    规则：
    - 中文连续单字合并为 bigram（"司马懿" -> ["司马", "马懿"]）
    - 英文/数字 token 按词，单字英文丢弃
    - 过滤 stopword / 高频功能字
    - 保留出现顺序与重复（调用方自行去重）
    """
    if not text:
        return []
    grams: list[str] = []
    raw = tokenize(text)
    cjk_buffer: list[str] = []

    def flush_cjk() -> None:
        if len(cjk_buffer) >= 2:
            for i in range(len(cjk_buffer) - 1):
                bg = cjk_buffer[i] + cjk_buffer[i + 1]
                if cjk_buffer[i] in CN_STOPCHARS and cjk_buffer[i + 1] in CN_STOPCHARS:
                    continue
                grams.append(bg)
        elif len(cjk_buffer) == 1:
            ch = cjk_buffer[0]
            if ch not in CN_STOPCHARS:
                grams.append(ch)
        cjk_buffer.clear()

    for tok in raw:
        if _is_cjk(tok):
            cjk_buffer.append(tok)
            continue
        flush_cjk()
        if len(tok) <= 1:
            continue
        if tok in EN_STOPWORDS:
            continue
        grams.append(tok)
    flush_cjk()
    return grams


def attribution_grams_set(text: str) -> set[str]:
    return set(attribution_grams(text))


def attribution_grams_counter(text: str) -> Counter:
    return Counter(attribution_grams(text))


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


# ---------- ISR / NAR (片段级溯源) ----------

@dataclass
class SourceAttribution:
    """答案信息的来源归因结果。

    n_answer_grams = 答案中有效溯源片段的"种类数"（去重后）。
    n_from_positive / n_from_negative / n_from_neither 之和 = n_answer_grams。
    同时命中两类时按"该片段在两类中的出现次数差"分配（pos_cnt > neg_cnt 归 pos，
    反之归 neg，相等则视为不可判别 → neither，防止 positive 单边占便宜）。
    """

    n_answer_grams: int
    n_from_positive: int
    n_from_negative: int
    n_from_neither: int
    isr: float
    nar: float

    # 兼容旧字段名
    @property
    def n_answer_tokens(self) -> int:
        return self.n_answer_grams


def attribute_answer(
    answer: str,
    docs: list[str],
    labels: list[str],
    *,
    min_token_len: int | None = None,
) -> SourceAttribution:
    """把答案归因到 positive / negative / 无 三类来源（片段级 + 频次比较）。

    `min_token_len` 参数保留仅为向后兼容，已不再使用——过滤策略全部走
    `attribution_grams` 的 stopword + bigram 逻辑。
    """
    _ = min_token_len  # 兼容签名

    ans_grams = set(attribution_grams(answer))
    if not ans_grams:
        return SourceAttribution(0, 0, 0, 0, 0.0, 0.0)

    pos_text = " ".join(d for d, l in zip(docs, labels) if l == "positive")
    neg_text = " ".join(d for d, l in zip(docs, labels) if l != "positive")
    pos_cnt = attribution_grams_counter(pos_text)
    neg_cnt = attribution_grams_counter(neg_text)

    n_pos = n_neg = n_neither = 0
    for g in ans_grams:
        cp = pos_cnt.get(g, 0)
        cn = neg_cnt.get(g, 0)
        if cp == 0 and cn == 0:
            n_neither += 1
        elif cp > cn:
            n_pos += 1
        elif cn > cp:
            n_neg += 1
        else:
            n_neither += 1

    total = len(ans_grams)
    return SourceAttribution(
        n_answer_grams=total,
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
