"""Build graph-style hard RAG data from MobileMem memories.

This generator is intentionally stricter than ``mobilemem_hard`` and
``mobilemem_reasoning``:

- every sample needs four positive memory documents;
- the answer is obtained by aggregating or comparing values across events;
- a single positive document is insufficient by construction;
- noise is borrowed from real, non-target MobileMem memories only.

The output stays RGB-compatible:

    {id, query, answer, positive, negative, positive_wrong, fakeanswer}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_hard_dataset import (  # noqa: E402
    ARTIFACT_LABEL,
    parent_id,
    parse_artifact,
)
from scripts.build_mobilemem_rag_dataset import (  # noqa: E402
    EventDoc,
    _contains_norm,
    _load_events,
    _norm_for_match,
    _open_mobilemem_root,
)
from scripts.build_mobilemem_reasoning_dataset import (  # noqa: E402
    format_seconds,
    parse_literal,
    parse_number,
    parse_timecode,
)


TARGET_POSITIVES = 4


@dataclass(frozen=True)
class Artifact:
    key: str
    event: EventDoc
    kind: str
    fields: dict[str, str]
    messages: tuple[tuple[str, str], ...]
    compact_doc: str
    parent_id: str
    parent_name: str

    @property
    def event_label(self) -> str:
        return f"{self.event.event_time[:16]}《{self.event.event_name}》"

    @property
    def month(self) -> str:
        return self.event.event_time[:7]

    @property
    def quarter(self) -> str:
        try:
            month = int(self.event.event_time[5:7])
        except ValueError:
            return "unknown"
        return f"{self.event.event_time[:4]}Q{(month - 1) // 3 + 1}"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str
    value_label: str
    entity_label: str
    entity_field: str | None
    unit: str


@dataclass(frozen=True)
class Candidate:
    artifacts: tuple[Artifact, ...]
    scope_type: str
    scope_desc: str
    operation: str
    metric: str
    metric_name: str
    answer: str
    query: str
    values: tuple[float, ...]
    answer_source_key: str | None = None
    chat_sender: str | None = None
    question_mode: str = "enumerated"
    target_condition: dict[str, Any] | None = None


METRICS: dict[str, MetricSpec] = {
    "shopping_price": MetricSpec(
        "shopping_price", "shopping", "实际价格", "商品名称", "item_name", "money"
    ),
    "money_amount": MetricSpec(
        "money_amount", "money", "转账金额", "收款方", "recipient_name", "money"
    ),
    "ticket_price": MetricSpec(
        "ticket_price", "ticket", "票价", "车次", "train_number", "money"
    ),
    "book_minutes": MetricSpec(
        "book_minutes", "book", "阅读时长折算分钟数", "书名", "title", "minutes"
    ),
    "music_remaining_seconds": MetricSpec(
        "music_remaining_seconds", "music", "剩余播放时长", "歌曲名", "song", "seconds"
    ),
    "video_like_gap": MetricSpec(
        "video_like_gap", "video", "点赞数比收藏数多出的次数", "视频标题", "title", "count"
    ),
    "video_like_danmaku_sum": MetricSpec(
        "video_like_danmaku_sum", "video", "点赞数和弹幕数合计", "视频标题", "title", "count"
    ),
    "friend_interactions": MetricSpec(
        "friend_interactions", "friend", "点赞和评论互动总次数", None, None, "count"
    ),
    "chat_sender_count": MetricSpec(
        "chat_sender_count", "chat", "目标成员发言条数", None, None, "messages"
    ),
}


def fmt_decimal(value: float, *, digits: int = 2) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def fmt_money(value: float) -> str:
    return f"{fmt_decimal(value)}元"


def fmt_count(value: float, unit: str = "次") -> str:
    return f"{int(round(value))}{unit}"


def fmt_minutes(value: float) -> str:
    return f"{int(round(value))}分钟"


def fmt_value(metric: str, value: float) -> str:
    spec = METRICS[metric]
    if spec.unit == "money":
        return fmt_money(value)
    if spec.unit == "minutes":
        return fmt_minutes(value)
    if spec.unit == "seconds":
        return format_seconds(int(round(value)))
    if spec.unit == "messages":
        return fmt_count(value, "条")
    return fmt_count(value, "次")


def is_integer_value(value: float) -> bool:
    return math.isclose(value, round(value), abs_tol=1e-9)


def parse_reading_minutes(value: str) -> float | None:
    text = str(value)
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*小时", text)
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*分(?:钟)?", text)
    total = 0.0
    matched = False
    if hour_match:
        total += float(hour_match.group(1)) * 60
        matched = True
    if minute_match:
        total += float(minute_match.group(1))
        matched = True
    if matched:
        return total
    number = parse_number(value)
    if number is None:
        return None
    return number


def metric_value(artifact: Artifact, metric: str, *, sender: str | None = None) -> float | None:
    fields = artifact.fields
    if metric == "shopping_price":
        return parse_number(fields.get("price", ""))
    if metric == "money_amount":
        return parse_number(fields.get("amount", ""))
    if metric == "ticket_price":
        return parse_number(fields.get("price", ""))
    if metric == "book_minutes":
        return parse_reading_minutes(fields.get("reading_time", ""))
    if metric == "music_remaining_seconds":
        duration = parse_timecode(fields.get("duration", ""))
        current = parse_timecode(fields.get("current_time", ""))
        if duration is None or current is None or duration <= current:
            return None
        return float(duration - current)
    if metric == "video_like_gap":
        like = parse_number(fields.get("like_count", ""))
        fav = parse_number(fields.get("fav_count", ""))
        if like is None or fav is None or like < fav:
            return None
        return like - fav
    if metric == "video_like_danmaku_sum":
        like = parse_number(fields.get("like_count", ""))
        danmaku = parse_number(fields.get("danmaku_count", ""))
        if like is None or danmaku is None:
            return None
        return like + danmaku
    if metric == "friend_interactions":
        likes = parse_literal(fields.get("likes", "[]"))
        comments = parse_literal(fields.get("comments", "[]"))
        if isinstance(likes, list) and isinstance(comments, list):
            return float(len(likes) + len(comments))
        return None
    if metric == "chat_sender_count":
        if not sender:
            return None
        counts = Counter(s for s, _ in artifact.messages if s != "关联事件")
        count = counts.get(sender, 0)
        return float(count) if count > 0 else None
    return None


def entity_value(artifact: Artifact, metric: str) -> str | None:
    field = METRICS[metric].entity_field
    if not field:
        return None
    value = str(artifact.fields.get(field, "")).strip()
    if not value or len(value) > 90:
        return None
    return value


def safe_metric_values(artifacts: Iterable[Artifact], metric: str, *, sender: str | None = None) -> list[float] | None:
    values: list[float] = []
    for artifact in artifacts:
        value = metric_value(artifact, metric, sender=sender)
        if value is None:
            return None
        values.append(value)
    return values


def compact_artifact_doc(event: EventDoc, raw_doc: str, idx: int) -> Artifact | None:
    kind, fields, messages = parse_artifact(raw_doc)
    if kind not in {spec.kind for spec in METRICS.values()}:
        return None
    clean_messages = tuple((s, t) for s, t in messages if s and t and s != "关联事件")
    p_id = parent_id(event)
    p_name = str(event.metadata.get("parent_event_name") or event.event_name)
    label = ARTIFACT_LABEL.get(kind, kind)

    lines = [
        "记忆证据: MobileMem 私人生活记录",
        f"事件ID: {event.event_id}",
        f"事件名称: {event.event_name}",
        f"发生时间: {event.event_time[:16]}",
        f"长期事件: {p_name}",
        f"记录类型: {label}",
    ]

    if kind == "shopping":
        price = parse_number(fields.get("price", ""))
        lines.extend(
            [
                f"商品名称: {fields.get('item_name', '')}",
                f"店铺: {fields.get('shop_name', '')}",
                f"实际价格: {fmt_money(price) if price is not None else fields.get('price', '')}",
                f"下单时间: {fields.get('order_time', '')}",
                f"评分: {fields.get('rating', '')}",
            ]
        )
    elif kind == "money":
        amount = parse_number(fields.get("amount", ""))
        lines.extend(
            [
                f"收款方: {fields.get('recipient_name', '')}",
                f"转账金额: {fmt_money(amount) if amount is not None else fields.get('amount', '')}",
                f"转账时间: {fields.get('transfer_time', '')}",
                f"到账时间: {fields.get('receive_time', '')}",
                f"交易说明: {fields.get('description', '')}",
            ]
        )
    elif kind == "ticket":
        price = parse_number(fields.get("price", ""))
        lines.extend(
            [
                f"车次: {fields.get('train_number', '')}",
                f"出发站: {fields.get('departure_station', '')}",
                f"到达站: {fields.get('arrival_station', '')}",
                f"座位: {fields.get('seat_number', '')}",
                f"票价: {fmt_money(price) if price is not None else fields.get('price', '')}",
            ]
        )
    elif kind == "book":
        minutes = parse_reading_minutes(fields.get("reading_time", ""))
        lines.extend(
            [
                f"书名: {fields.get('title', '')}",
                f"作者: {fields.get('author', '')}",
                f"阅读进度: {fields.get('progress', '')}",
                f"阅读时长: {fields.get('reading_time', '')}",
                f"评分: {fields.get('rating', '')}",
            ]
        )
    elif kind == "music":
        lines.extend(
            [
                f"歌曲名: {fields.get('song', '')}",
                f"歌手: {fields.get('artist', '')}",
                f"总时长: {fields.get('duration', '')}",
                f"当前播放: {fields.get('current_time', '')}",
                f"歌单: {fields.get('playlist', '')}",
            ]
        )
    elif kind == "video":
        lines.extend(
            [
                f"视频标题: {fields.get('title', '')}",
                f"UP主: {fields.get('uploader', '')}",
                f"点赞数: {fields.get('like_count', '')}",
                f"收藏数: {fields.get('fav_count', '')}",
                f"弹幕数: {fields.get('danmaku_count', '')}",
            ]
        )
    elif kind == "friend":
        lines.extend(
            [
                f"朋友圈正文: {fields.get('post_text', '')}",
                f"发布时间: {fields.get('post_time', '')}",
                f"点赞名单: {fields.get('likes', '')}",
                f"评论列表: {fields.get('comments', '')}",
            ]
        )
    elif kind == "chat":
        counts = Counter(s for s, _ in clean_messages)
        count_text = "；".join(f"{sender}={count}条" for sender, count in counts.most_common(10))
        excerpt = " / ".join(f"{s}: {t[:28]}" for s, t in clean_messages[:4])
        lines.extend(
            [
                f"群聊发言计数: {count_text}",
                f"群聊摘录: {excerpt}",
            ]
        )

    compact = "\n".join(line for line in lines if not line.endswith(": "))
    return Artifact(
        key=f"{event.uid}:{event.event_id}:{kind}:{idx}",
        event=event,
        kind=kind,
        fields=fields,
        messages=clean_messages,
        compact_doc=compact,
        parent_id=p_id,
        parent_name=p_name,
    )


def collect_artifacts(events: list[EventDoc]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for event in events:
        for idx, doc in enumerate(event.related_docs):
            artifact = compact_artifact_doc(event, doc, idx)
            if artifact is not None:
                artifacts.append(artifact)
    return artifacts


def dedupe_events(events: list[EventDoc]) -> list[EventDoc]:
    grouped: dict[str, list[EventDoc]] = defaultdict(list)
    for event in events:
        grouped[event.id_key].append(event)

    out: list[EventDoc] = []
    for values in grouped.values():
        values = sorted(
            values,
            key=lambda e: (
                e.metadata.get("source") == "stage4_5_sub_events",
                bool(e.metadata.get("parent_event_name")),
                len(e.related_docs),
            ),
            reverse=True,
        )
        chosen = values[0]
        merged_docs: list[str] = []
        seen: set[str] = set()
        for event in values:
            for doc in event.related_docs:
                key = _norm_for_match(doc)
                if key and key not in seen:
                    seen.add(key)
                    merged_docs.append(doc)
        chosen.related_docs = merged_docs
        out.append(chosen)
    return out


def event_list_text(artifacts: tuple[Artifact, ...]) -> str:
    return "、".join(a.event_label for a in artifacts)


def build_query(
    artifacts: tuple[Artifact, ...],
    *,
    scope_desc: str,
    operation: str,
    metric: str,
    sender: str | None = None,
) -> str:
    spec = METRICS[metric]
    labels = event_list_text(artifacts)
    prefix = (
        f"在{scope_desc}的私人记忆中，只统计以下{len(artifacts)}个事件：{labels}。"
        f"根据这些事件各自的{ARTIFACT_LABEL.get(spec.kind, spec.kind)}，"
    )
    if operation == "sum":
        if metric == "chat_sender_count":
            return prefix + f"{sender}在这些群聊里的发言条数合计是多少？"
        return prefix + f"{spec.value_label}合计是多少？"
    if operation == "range":
        if metric == "chat_sender_count":
            return prefix + f"{sender}单次群聊发言条数的最大值和最小值相差多少？"
        return prefix + f"{spec.value_label}的最大值和最小值相差多少？"
    if operation == "max_entity":
        return prefix + f"哪条记录的{spec.value_label}最大？请只回答该记录的{spec.entity_label}。"
    if operation == "min_entity":
        return prefix + f"哪条记录的{spec.value_label}最小？请只回答该记录的{spec.entity_label}。"
    raise ValueError(f"unknown operation: {operation}")


def build_conditioned_window_query(
    *,
    artifacts: tuple[Artifact, ...],
    operation: str,
    metric: str,
) -> tuple[str, dict[str, Any]]:
    spec = METRICS[metric]
    record_label = ARTIFACT_LABEL.get(spec.kind, spec.kind)
    start_time = artifacts[0].event.event_time[:16]
    end_time = artifacts[-1].event.event_time[:16]
    condition = {
        "mode": "conditioned_window",
        "record_type": record_label,
        "artifact_kind": spec.kind,
        "start_time": start_time,
        "end_time": end_time,
        "expected_target_records": len(artifacts),
        "selection_rule": (
            "Use only records whose MobileMem artifact type matches the requested "
            "type and whose event time is inside the closed time window."
        ),
    }
    prefix = (
        "在检索到的 MobileMem 私人记忆中，请先筛选目标记录，再计算答案。"
        f"目标记录必须同时满足：记录类型是“{record_label}”；"
        f"发生时间不早于 {start_time}，且不晚于 {end_time}；"
        f"共应筛出 {len(artifacts)} 条目标记录。"
        "不要统计时间窗外的记录，即使它们主题相近或记录类型相同。"
    )
    if operation == "sum":
        query = (
            f"{prefix}根据目标记录里的{spec.value_label}，把这 {len(artifacts)} 条目标记录的数值全部相加。"
            "不要只返回其中一条记录或部分记录的数值。最终合计是多少？只输出数值和单位。"
        )
    elif operation == "range":
        query = (
            f"{prefix}根据目标记录里的{spec.value_label}，最大值和最小值相差多少？"
            "只输出数值和单位。"
        )
    else:
        raise ValueError(f"conditioned_window does not support operation: {operation}")
    return query, condition


def make_candidate(
    artifacts: tuple[Artifact, ...],
    *,
    scope_type: str,
    scope_desc: str,
    operation: str,
    metric: str,
    sender: str | None = None,
) -> Candidate | None:
    if len(artifacts) != TARGET_POSITIVES:
        return None
    values = safe_metric_values(artifacts, metric, sender=sender)
    if values is None:
        return None
    if any(v < 0 for v in values):
        return None

    answer_source_key: str | None = None
    if operation == "sum":
        answer = fmt_value(metric, sum(values))
    elif operation == "range":
        diff = max(values) - min(values)
        if diff <= 0:
            return None
        answer = fmt_value(metric, diff)
    elif operation in {"max_entity", "min_entity"}:
        if metric == "chat_sender_count":
            return None
        entities = [entity_value(a, metric) for a in artifacts]
        if any(not entity for entity in entities):
            return None
        if len({_norm_for_match(str(entity)) for entity in entities}) != len(entities):
            return None
        target = max(values) if operation == "max_entity" else min(values)
        if values.count(target) != 1:
            return None
        target_artifact = artifacts[values.index(target)]
        entity = str(entities[values.index(target)])
        answer = entity
        answer_source_key = target_artifact.key
    else:
        return None

    query = build_query(
        artifacts,
        scope_desc=scope_desc,
        operation=operation,
        metric=metric,
        sender=sender,
    )
    if _contains_norm(query, answer):
        return None
    if operation in {"sum", "range"} and any(_contains_norm(a.compact_doc, answer) for a in artifacts):
        return None
    if len(answer) > 90 or len(_norm_for_match(answer)) < 2:
        return None
    return Candidate(
        artifacts=artifacts,
        scope_type=scope_type,
        scope_desc=scope_desc,
        operation=operation,
        metric=metric,
        metric_name=METRICS[metric].value_label,
        answer=answer,
        query=query,
        values=tuple(values),
        answer_source_key=answer_source_key,
        chat_sender=sender,
    )


def make_conditioned_window_candidate(
    artifacts: tuple[Artifact, ...],
    *,
    operation: str,
    metric: str,
) -> Candidate | None:
    if len(artifacts) != TARGET_POSITIVES:
        return None
    if operation not in {"sum", "range"}:
        return None
    values = safe_metric_values(artifacts, metric)
    if values is None or any(v < 0 for v in values):
        return None
    # Decimals and larger magnitudes are intentionally allowed: non-round,
    # varied values stop the model from guessing or mentally shortcutting, so it
    # must actually read every target document. Only an absurd-outlier guard
    # remains to reject parse artifacts.
    if operation == "sum" and sum(values) > 10**7:
        return None

    if operation == "sum":
        answer = fmt_value(metric, sum(values))
    elif operation == "range":
        diff = max(values) - min(values)
        if diff <= 0:
            return None
        answer = fmt_value(metric, diff)
    else:
        return None

    query, condition = build_conditioned_window_query(
        artifacts=artifacts,
        operation=operation,
        metric=metric,
    )
    if _contains_norm(query, answer):
        return None
    if operation in {"sum", "range"} and any(_contains_norm(a.compact_doc, answer) for a in artifacts):
        return None
    if len(answer) > 90 or len(_norm_for_match(answer)) < 2:
        return None

    return Candidate(
        artifacts=artifacts,
        scope_type="conditioned_window",
        scope_desc=f"{condition['start_time']} 至 {condition['end_time']} 的{condition['record_type']}",
        operation=operation,
        metric=metric,
        metric_name=METRICS[metric].value_label,
        answer=answer,
        query=query,
        values=tuple(values),
        question_mode="conditioned_window",
        target_condition=condition,
    )


def four_doc_subsets(items: list[Artifact], rng: random.Random, *, max_subsets: int = 18) -> list[tuple[Artifact, ...]]:
    ordered = sorted(items, key=lambda a: (a.event.event_time, a.event.event_id, a.kind))
    out: list[tuple[Artifact, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(subset: Iterable[Artifact]) -> None:
        value = tuple(subset)
        if len(value) != TARGET_POSITIVES:
            return
        if len({a.event.event_id for a in value}) != TARGET_POSITIVES:
            return
        if len({_norm_for_match(a.compact_doc) for a in value}) != TARGET_POSITIVES:
            return
        key = tuple(a.key for a in value)
        if key in seen:
            return
        seen.add(key)
        out.append(value)

    for i in range(0, max(0, len(ordered) - TARGET_POSITIVES + 1)):
        add(ordered[i : i + TARGET_POSITIVES])
        if len(out) >= max_subsets // 2:
            break

    if len(ordered) <= 14:
        combos = list(combinations(ordered, TARGET_POSITIVES))
        rng.shuffle(combos)
        for combo in combos:
            add(combo)
            if len(out) >= max_subsets:
                break
    else:
        attempts = 0
        while len(out) < max_subsets and attempts < max_subsets * 12:
            attempts += 1
            add(sorted(rng.sample(ordered, TARGET_POSITIVES), key=lambda a: a.event.event_time))
    return out


def collect_candidates(artifacts: list[Artifact], rng: random.Random) -> list[Candidate]:
    candidates: list[Candidate] = []
    metrics_by_kind: dict[str, list[str]] = defaultdict(list)
    for name, spec in METRICS.items():
        metrics_by_kind[spec.kind].append(name)

    grouped: list[tuple[str, str, list[Artifact]]] = []
    by_parent_kind: dict[tuple[str, str], list[Artifact]] = defaultdict(list)
    by_month_kind: dict[tuple[str, str], list[Artifact]] = defaultdict(list)
    by_quarter_kind: dict[tuple[str, str], list[Artifact]] = defaultdict(list)

    for artifact in artifacts:
        by_parent_kind[(artifact.parent_id, artifact.kind)].append(artifact)
        by_month_kind[(artifact.month, artifact.kind)].append(artifact)
        by_quarter_kind[(artifact.quarter, artifact.kind)].append(artifact)

    for (p_id, kind), items in by_parent_kind.items():
        if len(items) >= TARGET_POSITIVES:
            parent_name = items[0].parent_name
            grouped.append(("parent", f"长期事件“{parent_name}”", items))
    for (month, kind), items in by_month_kind.items():
        if len(items) >= TARGET_POSITIVES:
            grouped.append(("month", f"{month.replace('-', '年')}月", items))
    for (quarter, kind), items in by_quarter_kind.items():
        if len(items) >= TARGET_POSITIVES:
            year, q = quarter.split("Q", 1)
            grouped.append(("quarter", f"{year}年第{q}季度", items))

    for scope_type, scope_desc, items in grouped:
        kind = items[0].kind
        for subset in four_doc_subsets(items, rng):
            for metric in metrics_by_kind[kind]:
                if metric == "chat_sender_count":
                    senders = Counter()
                    for artifact in subset:
                        senders.update(s for s, _ in artifact.messages if s != "关联事件")
                    common_senders = [
                        sender
                        for sender, _ in senders.most_common(5)
                        if all(metric_value(a, metric, sender=sender) is not None for a in subset)
                    ]
                    for sender in common_senders[:2]:
                        for operation in ("sum", "range"):
                            cand = make_candidate(
                                subset,
                                scope_type=scope_type,
                                scope_desc=scope_desc,
                                operation=operation,
                                metric=metric,
                                sender=sender,
                            )
                            if cand:
                                candidates.append(cand)
                    continue

                if safe_metric_values(subset, metric) is None:
                    continue
                operations = ["sum", "range"]
                if METRICS[metric].entity_field:
                    operations.extend(["max_entity", "min_entity"])
                for operation in operations:
                    cand = make_candidate(
                        subset,
                        scope_type=scope_type,
                        scope_desc=scope_desc,
                        operation=operation,
                        metric=metric,
                    )
                    if cand:
                        candidates.append(cand)

    return candidates


def _minute_key(artifact: Artifact) -> str:
    """Event time at the minute precision the model actually sees in docs."""
    return artifact.event.event_time[:16]


def window_is_unique(
    subset: tuple[Artifact, ...], same_kind: list[Artifact]
) -> bool:
    """True iff the 4 positives are the ONLY same-type records inside the closed
    [first, last] event-time window.

    The conditioned_window question tells the model to select every record of a
    given type within a time window and aggregate them. If any other same-type
    record falls inside that window, the stated filter would match more than the
    4 positives and the gold answer becomes ambiguous (it silently punishes a
    model that correctly reads all docs). Comparison uses minute precision to
    match the timestamps rendered in the documents.
    """
    start = _minute_key(subset[0])
    end = _minute_key(subset[-1])
    positive_keys = {a.key for a in subset}
    for artifact in same_kind:
        if artifact.key in positive_keys:
            continue
        if start <= _minute_key(artifact) <= end:
            return False
    return True


def collect_conditioned_window_candidates(
    artifacts: list[Artifact],
    rng: random.Random,
    *,
    operations: set[str] | None = None,
    metrics: set[str] | None = None,
) -> list[Candidate]:
    """Build source-constrained hard samples without behavior filtering.

    Each candidate uses four consecutive same-type records as the positive
    window. The question defines the target set by record type plus a closed
    event-time window, so clean context is directly answerable while noisy
    context requires support filtering instead of aggregating every retrieved
    same-type record.
    """
    del rng  # deterministic source ordering is part of this generation mode.
    allowed_ops = operations or {"sum"}
    allowed_metrics = metrics or {"money_amount", "shopping_price"}
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)

    for metric in sorted(allowed_metrics):
        spec = METRICS.get(metric)
        if spec is None:
            continue
        if spec.kind not in {"money", "shopping"}:
            continue
        # Full same-type universe defines the query's filter set; uniqueness must
        # hold over ALL of it (including decimal/unparseable records), not just
        # the integer-valued ones.
        same_kind = sorted(
            [a for a in artifacts if a.kind == spec.kind],
            key=lambda a: (a.event.event_time, a.event.event_id, a.key),
        )
        # Candidate positives just need a usable numeric value; decimals are now
        # allowed (see make_conditioned_window_candidate) to keep values varied.
        rows = sorted(
            [a for a in same_kind if metric_value(a, metric) is not None],
            key=lambda a: (a.event.event_time, a.event.event_id, a.key),
        )
        for start in range(0, len(rows) - TARGET_POSITIVES + 1):
            subset = tuple(rows[start : start + TARGET_POSITIVES])
            if len({a.event.event_id for a in subset}) != TARGET_POSITIVES:
                continue
            if not window_is_unique(subset, same_kind):
                continue
            for operation in sorted(allowed_ops):
                cand = make_conditioned_window_candidate(
                    subset,
                    operation=operation,
                    metric=metric,
                )
                if cand:
                    groups[(metric, operation)].append(cand)

    ordered: list[Candidate] = []
    keys = sorted(groups)
    while any(groups.values()):
        for key in keys:
            values = groups[key]
            if values:
                ordered.append(values.pop(0))
    return ordered


def candidate_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.question_mode,
        tuple(a.key for a in candidate.artifacts),
        candidate.operation,
        candidate.metric,
        candidate.chat_sender,
    )


def balanced_candidates(candidates: list[Candidate], rng: random.Random) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.metric].append(cand)
    for values in groups.values():
        rng.shuffle(values)

    ordered: list[Candidate] = []
    keys = sorted(groups)
    while any(groups.values()):
        for key in keys:
            values = groups[key]
            if values:
                ordered.append(values.pop())
    return ordered


def doc_leaks_answer(doc: str, answer: str) -> bool:
    return _contains_norm(doc, answer)


def choose_negatives(
    candidate: Candidate,
    artifacts: list[Artifact],
    *,
    count: int,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]]]:
    positive_keys = {a.key for a in candidate.artifacts}
    positive_parent_ids = {a.parent_id for a in candidate.artifacts}
    positive_months = {a.month for a in candidate.artifacts}
    positive_event_ids = {a.event.event_id for a in candidate.artifacts}
    target_kind = METRICS[candidate.metric].kind

    rows: list[tuple[int, float, float, Artifact]] = []
    query_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", candidate.query.lower()))
    ordered_same_kind = sorted(
        [a for a in artifacts if a.kind == target_kind],
        key=lambda a: (a.event.event_time, a.event.event_id, a.key),
    )
    same_kind_pos = {a.key: idx for idx, a in enumerate(ordered_same_kind)}
    positive_positions = [same_kind_pos[a.key] for a in candidate.artifacts if a.key in same_kind_pos]

    # Conditioned-window correctness + difficulty controls.
    cw_start = cw_end = None
    forced_keys: set[str] = set()
    if candidate.question_mode == "conditioned_window" and candidate.target_condition:
        cw_start = candidate.target_condition.get("start_time")
        cw_end = candidate.target_condition.get("end_time")
        # Fix 3: guarantee same-type "just outside the window" distractors so the
        # time filter is load-bearing (the model must compare timestamps, not
        # only the record type). Take the nearest out-of-window same-type records
        # on each side of the window.
        eligible = [
            a
            for a in ordered_same_kind
            if a.key not in positive_keys
            and a.event.event_id not in positive_event_ids
            and not doc_leaks_answer(a.compact_doc, candidate.answer)
        ]
        before = [a for a in eligible if cw_start and _minute_key(a) < cw_start]
        after = [a for a in eligible if cw_end and _minute_key(a) > cw_end]
        for artifact in before[-2:]:
            forced_keys.add(artifact.key)
        for artifact in after[:2]:
            forced_keys.add(artifact.key)

    for artifact in artifacts:
        if artifact.key in positive_keys:
            continue
        if artifact.event.event_id in positive_event_ids:
            continue
        if doc_leaks_answer(artifact.compact_doc, candidate.answer):
            continue
        # Fix 2: a same-type record inside the window would satisfy the stated
        # filter, so it must never be a negative (it would corrupt the gold).
        if (
            cw_start is not None
            and artifact.kind == target_kind
            and cw_start <= _minute_key(artifact) <= cw_end
        ):
            continue
        doc_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", artifact.compact_doc.lower()))
        score = float(len(query_tokens & doc_tokens))
        if artifact.kind == target_kind:
            score += 30
        if artifact.parent_id in positive_parent_ids:
            score += 16
        if artifact.month in positive_months:
            score += 8
        if candidate.chat_sender and _contains_norm(artifact.compact_doc, candidate.chat_sender):
            score += 6
        if candidate.question_mode == "conditioned_window" and artifact.kind == target_kind:
            idx = same_kind_pos.get(artifact.key)
            if idx is not None and positive_positions:
                distance = min(abs(idx - pidx) for pidx in positive_positions)
                if distance <= 8:
                    score += 70 - distance * 6
            value = metric_value(artifact, candidate.metric, sender=candidate.chat_sender)
            if value is not None and candidate.values:
                nearest = min(abs(value - v) for v in candidate.values)
                scale = max(max(abs(v) for v in candidate.values), 1.0)
                score += max(0.0, 14.0 - 14.0 * nearest / scale)
        priority = 1 if artifact.key in forced_keys else 0
        rows.append((priority, score, rng.random(), artifact))

    rows.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    docs: list[str] = []
    meta: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _priority, score, _, artifact in rows:
        doc_norm = _norm_for_match(artifact.compact_doc)
        if not doc_norm or doc_norm in seen:
            continue
        seen.add(doc_norm)
        docs.append(artifact.compact_doc)
        meta.append(
            {
                "source_event_id": artifact.event.event_id,
                "source_event_name": artifact.event.event_name,
                "source_artifact_type": artifact.kind,
                "same_parent_as_target": artifact.parent_id in positive_parent_ids,
                "same_month_as_target": artifact.month in positive_months,
                "score": round(score, 3),
            }
        )
        if len(docs) >= count:
            break
    if len(docs) < count:
        raise ValueError(f"only {len(docs)}/{count} negatives available")
    return docs, meta


def make_row(
    candidate: Candidate,
    artifacts: list[Artifact],
    *,
    sample_id: int,
    negative_docs: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    positives = [a.compact_doc for a in candidate.artifacts]
    if _contains_norm(candidate.query, candidate.answer):
        return None
    if candidate.operation in {"sum", "range"} and any(doc_leaks_answer(doc, candidate.answer) for doc in positives):
        return None
    negatives, negative_meta = choose_negatives(
        candidate,
        artifacts,
        count=negative_docs,
        rng=rng,
    )
    support_path = []
    for artifact, value in zip(candidate.artifacts, candidate.values):
        support_path.append(
            {
                "event_id": artifact.event.event_id,
                "event_name": artifact.event.event_name,
                "event_time": artifact.event.event_time,
                "artifact_type": artifact.kind,
                "metric_value": fmt_value(candidate.metric, value),
                "parent_event_id": artifact.parent_id,
                "parent_event_name": artifact.parent_name,
            }
        )

    return {
        "id": sample_id,
        "query": candidate.query,
        "answer": [candidate.answer],
        "positive": positives,
        "negative": negatives,
        "positive_wrong": [],
        "fakeanswer": "",
        "mobilemem_meta": {
            "uid": candidate.artifacts[0].event.uid,
            "event_ids": [a.event.event_id for a in candidate.artifacts],
            "event_names": [a.event.event_name for a in candidate.artifacts],
            "event_times": [a.event.event_time for a in candidate.artifacts],
            "parent_event_ids": sorted({a.parent_id for a in candidate.artifacts}),
            "parent_event_names": sorted({a.parent_name for a in candidate.artifacts}),
            "noise_policy": "real_cross_event_non_contradictory_non_supporting",
            "hardness": "graph_multi_event_multi_document",
            "question_mode": candidate.question_mode,
            "target_condition": candidate.target_condition,
            "scope_type": candidate.scope_type,
            "scope_desc": candidate.scope_desc,
            "operation": candidate.operation,
            "metric": candidate.metric,
            "metric_name": candidate.metric_name,
            "answer_artifact_type": METRICS[candidate.metric].kind,
            "answer_source_key": candidate.answer_source_key,
            "chat_sender": candidate.chat_sender,
            "required_positive_docs": len(candidate.artifacts),
            "single_doc_answerable": False,
            "answer_is_derived": candidate.operation in {"sum", "range"},
            "requires_cross_event_comparison": True,
            "support_path": support_path,
            "target_negative_docs": negative_docs,
            "actual_negative_docs": len(negatives),
            "note": (
                "The question is scoped to four target events. Negative documents are "
                "real memories from other events and do not contradict the target records."
            ),
        },
        "negative_meta": negative_meta,
    }


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    operation_counts = Counter((r.get("mobilemem_meta") or {}).get("operation") for r in rows)
    metric_counts = Counter((r.get("mobilemem_meta") or {}).get("metric") for r in rows)
    scope_counts = Counter((r.get("mobilemem_meta") or {}).get("scope_type") for r in rows)
    leaks = []
    weak = []
    for row in rows:
        answer = str((row.get("answer") or [""])[0])
        if _contains_norm(row.get("query", ""), answer):
            leaks.append({"id": row.get("id"), "where": "query"})
        if any(_contains_norm(doc, answer) for doc in row.get("negative") or []):
            leaks.append({"id": row.get("id"), "where": "negative"})
        meta = row.get("mobilemem_meta") or {}
        if len(row.get("positive") or []) != TARGET_POSITIVES:
            weak.append({"id": row.get("id"), "reason": "positive_count"})
        if meta.get("single_doc_answerable") is not False:
            weak.append({"id": row.get("id"), "reason": "single_doc_flag"})
        if meta.get("required_positive_docs") != TARGET_POSITIVES:
            weak.append({"id": row.get("id"), "reason": "required_positive_docs"})
    return {
        "rows": len(rows),
        "positive_docs_each": TARGET_POSITIVES,
        "min_negatives": min((len(r.get("negative") or []) for r in rows), default=0),
        "max_negatives": max((len(r.get("negative") or []) for r in rows), default=0),
        "operation_counts": dict(sorted(operation_counts.items())),
        "metric_counts": dict(sorted(metric_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "leak_count": len(leaks),
        "weak_count": len(weak),
        "leak_examples": leaks[:5],
        "weak_examples": weak[:5],
    }


def build_graph_hard_dataset(
    *,
    input_path: Path,
    output_path: Path,
    language: str,
    limit: int,
    seed: int,
    start_id: int,
    negative_docs: int,
    allowed_operations: set[str] | None = None,
    allowed_metrics: set[str] | None = None,
    question_mode: str = "enumerated",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir, tmp = _open_mobilemem_root(input_path)
    try:
        events = dedupe_events([e for e in _load_events(data_dir) if e.language == language])
        rng = random.Random(seed)
        artifacts = collect_artifacts(events)
        if question_mode == "conditioned_window":
            candidates = collect_conditioned_window_candidates(
                artifacts,
                rng,
                operations=allowed_operations,
                metrics=allowed_metrics,
            )
        elif question_mode == "enumerated":
            candidates = balanced_candidates(collect_candidates(artifacts, rng), rng)
        else:
            raise ValueError(f"unknown question_mode: {question_mode}")

        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for candidate in candidates:
            if limit > 0 and len(rows) >= limit:
                break
            if allowed_operations and candidate.operation not in allowed_operations:
                continue
            if allowed_metrics and candidate.metric not in allowed_metrics:
                continue
            key = candidate_key(candidate)
            if key in seen:
                continue
            try:
                row = make_row(
                    candidate,
                    artifacts,
                    sample_id=start_id + len(rows),
                    negative_docs=negative_docs,
                    rng=rng,
                )
            except ValueError:
                continue
            if not row:
                continue
            seen.add(key)
            rows.append(row)
    finally:
        if tmp is not None:
            tmp.cleanup()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = audit_rows(rows)
    audit_path = output_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT.parent / "data" / "0418(1).zip",
        help="MobileMem zip, extracted root, or data/ directory.",
    )
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-id", type=int, default=340000)
    parser.add_argument("--negative-docs", type=int, default=76)
    parser.add_argument(
        "--allow-operation",
        action="append",
        default=[],
        help="Restrict generated candidates to this operation. Repeatable.",
    )
    parser.add_argument(
        "--allow-metric",
        action="append",
        default=[],
        help="Restrict generated candidates to this metric. Repeatable.",
    )
    parser.add_argument(
        "--question-mode",
        choices=("enumerated", "conditioned_window"),
        default="enumerated",
        help="Question construction mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to data/rgb/<language>_mobilemem_graph_hard.json.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = ROOT / "data" / "rgb" / f"{args.language}_mobilemem_graph_hard.json"
    elif not output.is_absolute():
        output = ROOT / output

    rows, summary = build_graph_hard_dataset(
        input_path=args.input,
        output_path=output,
        language=args.language,
        limit=args.limit,
        seed=args.seed,
        start_id=args.start_id,
        negative_docs=args.negative_docs,
        allowed_operations=set(args.allow_operation) or None,
        allowed_metrics=set(args.allow_metric) or None,
        question_mode=args.question_mode,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "audit": str(output.with_suffix(".audit.json")),
                "rows": len(rows),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
