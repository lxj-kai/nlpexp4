"""Build MobileMem-Reasoning data with derived, non-verbatim answers.

This dataset is harder than ``mobilemem_hard``: it still requires an anchor
document plus an artifact document, but the gold answer is computed from fields
inside the artifact rather than copied from a visible field.

Examples:
- music duration - current_time -> remaining playback time
- transfer_time to receive_time -> arrival delay
- like_count - fav_count -> engagement gap
- chat messages -> sender message count

The output stays RGB-compatible:

    {id, query, answer, positive, negative, positive_wrong, fakeanswer}
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_hard_dataset import (  # noqa: E402
    ARTIFACT_LABEL,
    all_event_docs,
    anchor_clue,
    parent_id,
    parse_artifact,
)
from scripts.build_mobilemem_rag_dataset import (  # noqa: E402
    EventDoc,
    _contains_norm,
    _load_events,
    _norm_for_match,
    _open_mobilemem_root,
    _similarity_key,
)


@dataclass(frozen=True)
class ReasoningCandidate:
    event: EventDoc
    answer_doc: str
    answer: str
    question_tail: str
    artifact_type: str
    answer_field: str
    reasoning_type: str
    support_fields: tuple[str, ...]


def parse_timecode(value: str) -> int | None:
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        if seconds >= 60:
            return None
        return minutes * 60 + seconds
    hours, minutes, seconds = nums
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def format_seconds(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}分{sec:02d}秒"
    hours, rem = divmod(minutes, 60)
    return f"{hours}小时{rem:02d}分{sec:02d}秒"


def parse_number(value: str) -> float | None:
    text = str(value).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    num = float(m.group(0))
    if "万" in text:
        num *= 10000
    return num


def parse_datetime(value: str) -> datetime | None:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_literal(value: str) -> Any | None:
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def format_decimal(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def valid_reasoned_answer(answer: object, event: EventDoc, answer_doc: str) -> str | None:
    text = str(answer).strip()
    if not text:
        return None
    if len(text) > 40:
        return None
    if len(_norm_for_match(text)) < 2:
        return None
    if text in {"0秒", "1秒", "2秒", "3秒", "4秒", "0条", "1条", "0次", "1次"}:
        return None
    if _contains_norm(event.text, text):
        return None
    if _contains_norm(answer_doc, text):
        return None
    return text


def make_candidate(
    *,
    event: EventDoc,
    doc: str,
    answer: object,
    question_tail: str,
    artifact_type: str,
    answer_field: str,
    reasoning_type: str,
    support_fields: tuple[str, ...],
) -> ReasoningCandidate | None:
    valid = valid_reasoned_answer(answer, event, doc)
    if not valid:
        return None
    return ReasoningCandidate(
        event=event,
        answer_doc=doc,
        answer=valid,
        question_tail=question_tail,
        artifact_type=artifact_type,
        answer_field=answer_field,
        reasoning_type=reasoning_type,
        support_fields=support_fields,
    )


def build_reasoning_candidates(event: EventDoc, doc: str) -> list[ReasoningCandidate]:
    kind, fields, messages = parse_artifact(doc)
    out: list[ReasoningCandidate] = []

    if kind == "music":
        duration = parse_timecode(fields.get("duration", ""))
        current = parse_timecode(fields.get("current_time", ""))
        if duration is not None and current is not None and duration > current:
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=format_seconds(duration - current),
                question_tail="这首歌从当前播放时间到结束还剩多长时间？",
                artifact_type=kind,
                answer_field="duration_minus_current_time",
                reasoning_type="time_remaining",
                support_fields=("duration", "current_time"),
            )
            if cand:
                out.append(cand)

    elif kind == "video":
        like = parse_number(fields.get("like_count", ""))
        fav = parse_number(fields.get("fav_count", ""))
        danmaku = parse_number(fields.get("danmaku_count", ""))
        if like is not None and fav is not None and like > fav:
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=str(int(round(like - fav))),
                question_tail="视频的点赞数比收藏数多多少？",
                artifact_type=kind,
                answer_field="like_count_minus_fav_count",
                reasoning_type="numeric_difference",
                support_fields=("like_count", "fav_count"),
            )
            if cand:
                out.append(cand)
        if like is not None and danmaku is not None:
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=str(int(round(like + danmaku))),
                question_tail="视频的点赞数和弹幕数合计是多少？",
                artifact_type=kind,
                answer_field="like_count_plus_danmaku_count",
                reasoning_type="numeric_sum",
                support_fields=("like_count", "danmaku_count"),
            )
            if cand:
                out.append(cand)

    elif kind == "money":
        start = parse_datetime(fields.get("transfer_time", ""))
        end = parse_datetime(fields.get("receive_time", ""))
        if start is not None and end is not None and end >= start:
            seconds = int((end - start).total_seconds())
            if 5 <= seconds <= 3600:
                cand = make_candidate(
                    event=event,
                    doc=doc,
                    answer=format_seconds(seconds),
                    question_tail="从转账时间到到账时间间隔多久？",
                    artifact_type=kind,
                    answer_field="receive_delay",
                    reasoning_type="time_delta",
                    support_fields=("transfer_time", "receive_time"),
                )
                if cand:
                    out.append(cand)

    elif kind == "ticket":
        # Synthetic passenger IDs are often templated, so birthdate extraction
        # creates many accidentally supporting distractors. Skip ticket-derived
        # answers until the source data has more identity diversity.
        pass

    elif kind == "shopping":
        price = parse_number(fields.get("price", ""))
        rating = parse_number(fields.get("rating", ""))
        if price is not None and rating is not None and rating > 0:
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=f"{format_decimal(price / rating)}元/星",
                question_tail="按实际价格除以评分计算，每颗星约对应多少钱？",
                artifact_type=kind,
                answer_field="price_div_rating",
                reasoning_type="numeric_ratio",
                support_fields=("price", "rating"),
            )
            if cand:
                out.append(cand)

    elif kind == "book":
        reading_time = fields.get("reading_time", "")
        hours = parse_number(reading_time)
        if hours is not None:
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=f"{format_decimal(hours * 60)}分钟",
                question_tail="读书记录里的阅读时长折算成分钟是多少？",
                artifact_type=kind,
                answer_field="reading_time_minutes",
                reasoning_type="unit_conversion",
                support_fields=("reading_time",),
            )
            if cand:
                out.append(cand)

    elif kind == "friend":
        likes = parse_literal(fields.get("likes", "[]"))
        comments = parse_literal(fields.get("comments", "[]"))
        if isinstance(likes, list) and isinstance(comments, list):
            total = len(likes) + len(comments)
            if total >= 2:
                cand = make_candidate(
                    event=event,
                    doc=doc,
                    answer=f"{total}次",
                    question_tail="朋友圈里的点赞和评论加起来一共有多少次互动？",
                    artifact_type=kind,
                    answer_field="likes_plus_comments",
                    reasoning_type="interaction_count",
                    support_fields=("likes", "comments"),
                )
                if cand:
                    out.append(cand)

    elif kind == "chat":
        counts = Counter(sender for sender, _ in messages)
        for sender, count in counts.most_common(2):
            if count < 2:
                continue
            cand = make_candidate(
                event=event,
                doc=doc,
                answer=f"{count}条",
                question_tail=f"相关群聊里，{sender}一共发了多少条消息？",
                artifact_type=kind,
                answer_field=f"{sender}_message_count",
                reasoning_type="chat_message_count",
                support_fields=("messages", sender),
            )
            if cand:
                out.append(cand)

    return out


def derived_answers_for_doc(
    doc: str,
    *,
    reasoning_type: str,
    answer_field: str,
    artifact_type: str,
) -> set[str]:
    kind, fields, messages = parse_artifact(doc)
    if kind != artifact_type:
        return set()

    out: set[str] = set()
    if reasoning_type == "time_remaining":
        duration = parse_timecode(fields.get("duration", ""))
        current = parse_timecode(fields.get("current_time", ""))
        if duration is not None and current is not None and duration > current:
            out.add(format_seconds(duration - current))
    elif reasoning_type == "numeric_difference":
        like = parse_number(fields.get("like_count", ""))
        fav = parse_number(fields.get("fav_count", ""))
        if like is not None and fav is not None and like > fav:
            out.add(str(int(round(like - fav))))
    elif reasoning_type == "numeric_sum":
        like = parse_number(fields.get("like_count", ""))
        danmaku = parse_number(fields.get("danmaku_count", ""))
        if like is not None and danmaku is not None:
            out.add(str(int(round(like + danmaku))))
    elif reasoning_type == "time_delta":
        start = parse_datetime(fields.get("transfer_time", ""))
        end = parse_datetime(fields.get("receive_time", ""))
        if start is not None and end is not None and end >= start:
            seconds = int((end - start).total_seconds())
            if 1 <= seconds <= 3600:
                out.add(format_seconds(seconds))
    elif reasoning_type == "numeric_ratio":
        price = parse_number(fields.get("price", ""))
        rating = parse_number(fields.get("rating", ""))
        if price is not None and rating is not None and rating > 0:
            out.add(f"{format_decimal(price / rating)}元/星")
    elif reasoning_type == "unit_conversion":
        hours = parse_number(fields.get("reading_time", ""))
        if hours is not None:
            out.add(f"{format_decimal(hours * 60)}分钟")
    elif reasoning_type == "interaction_count":
        likes = parse_literal(fields.get("likes", "[]"))
        comments = parse_literal(fields.get("comments", "[]"))
        if isinstance(likes, list) and isinstance(comments, list):
            total = len(likes) + len(comments)
            if total >= 2:
                out.add(f"{total}次")
    elif reasoning_type == "chat_message_count":
        counts = Counter(sender for sender, _ in messages)
        sender = answer_field.removesuffix("_message_count")
        if sender in counts:
            out.add(f"{counts[sender]}条")
    return out


def doc_derives_answer(
    doc: str,
    *,
    answer: str,
    reasoning_type: str,
    answer_field: str,
    artifact_type: str,
) -> bool:
    derived = derived_answers_for_doc(
        doc,
        reasoning_type=reasoning_type,
        answer_field=answer_field,
        artifact_type=artifact_type,
    )
    return any(_norm_for_match(x) == _norm_for_match(answer) for x in derived)


def choose_reasoning_negatives(
    candidate: ReasoningCandidate,
    events: list[EventDoc],
    *,
    count: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    target = candidate.event
    target_tokens = _similarity_key(target)
    rows: list[tuple[int, EventDoc, str, str]] = []
    for other in events:
        if other.id_key == target.id_key or other.language != target.language:
            continue
        overlap = len(target_tokens & _similarity_key(other))
        score = overlap
        if other.uid == target.uid:
            score += 5
        if parent_id(other) == parent_id(target):
            score += 16
        for doc, role in all_event_docs(other):
            role_score = score
            if role == candidate.artifact_type:
                role_score += 6
            if role == "event":
                role_score += 2
            rows.append((role_score, other, doc, role))

    rows.sort(key=lambda x: x[0], reverse=True)
    docs: list[str] = []
    meta: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, source, doc, role in rows:
        doc_norm = _norm_for_match(doc)
        if not doc_norm or doc_norm in seen:
            continue
        if _contains_norm(doc, candidate.answer):
            continue
        if _contains_norm(doc, target.event_name):
            continue
        if doc_derives_answer(
            doc,
            answer=candidate.answer,
            reasoning_type=candidate.reasoning_type,
            answer_field=candidate.answer_field,
            artifact_type=candidate.artifact_type,
        ):
            continue
        docs.append(doc)
        seen.add(doc_norm)
        meta.append(
            {
                "source_event_id": source.event_id,
                "source_event_name": source.event_name,
                "source_doc_role": role,
                "score": score,
                "same_parent": parent_id(source) == parent_id(target),
            }
        )
        if len(docs) >= count:
            break
    if len(docs) < count:
        raise ValueError(f"only {len(docs)}/{count} negatives available for {target.id_key}")
    return docs, meta


def candidate_key(c: ReasoningCandidate) -> tuple[str, str, str, str]:
    return c.event.id_key, c.artifact_type, c.answer_field, c.reasoning_type


def build_query(candidate: ReasoningCandidate) -> str | None:
    clue = anchor_clue(candidate.event, candidate.answer)
    if not clue:
        return None
    label = ARTIFACT_LABEL.get(candidate.artifact_type, "关联记录")
    query = (
        f"在私人记忆中，先根据“{clue}”定位对应事件；"
        f"再查看同一事件的{label}并进行计算，{candidate.question_tail}"
    )
    if _contains_norm(query, candidate.answer):
        return None
    return query


def balanced_candidates(
    candidates: list[ReasoningCandidate],
    rng: random.Random,
) -> list[ReasoningCandidate]:
    groups: dict[str, list[ReasoningCandidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.reasoning_type].append(cand)
    for values in groups.values():
        rng.shuffle(values)

    type_order = sorted(groups)
    ordered: list[ReasoningCandidate] = []
    used_events: set[str] = set()

    while True:
        progressed = False
        for typ in type_order:
            values = groups[typ]
            for idx, cand in enumerate(values):
                if cand.event.id_key in used_events:
                    continue
                ordered.append(cand)
                used_events.add(cand.event.id_key)
                values.pop(idx)
                progressed = True
                break
        if not progressed:
            break

    while any(groups.values()):
        for typ in type_order:
            values = groups[typ]
            if values:
                ordered.append(values.pop())

    return ordered


def make_sample(
    candidate: ReasoningCandidate,
    events: list[EventDoc],
    *,
    sample_id: int,
    negative_docs: int,
) -> dict[str, Any] | None:
    query = build_query(candidate)
    if not query:
        return None
    positives = [candidate.event.text, candidate.answer_doc]
    if any(_contains_norm(doc, candidate.answer) for doc in positives):
        return None
    if _contains_norm(query, candidate.answer):
        return None
    negatives, negative_meta = choose_reasoning_negatives(candidate, events, count=negative_docs)
    if any(_contains_norm(doc, candidate.answer) for doc in negatives):
        return None
    return {
        "id": sample_id,
        "query": query,
        "answer": [candidate.answer],
        "positive": positives,
        "negative": negatives,
        "positive_wrong": [],
        "fakeanswer": "",
        "mobilemem_meta": {
            "uid": candidate.event.uid,
            "event_id": candidate.event.event_id,
            "event_name": candidate.event.event_name,
            "event_time": candidate.event.event_time,
            "source": candidate.event.metadata.get("source"),
            "parent_event_id": candidate.event.metadata.get("parent_event_id"),
            "parent_event_name": candidate.event.metadata.get("parent_event_name"),
            "noise_policy": "semantic_near_event_non_contradictory",
            "hardness": "anchor_plus_artifact_derived_answer",
            "answer_artifact_type": candidate.artifact_type,
            "answer_field": candidate.answer_field,
            "reasoning_type": candidate.reasoning_type,
            "support_fields": list(candidate.support_fields),
            "requires_anchor_doc": True,
            "requires_artifact_doc": True,
            "answer_is_derived": True,
            "answer_not_verbatim_in_context": True,
            "negative_docs": negative_docs,
        },
        "negative_meta": negative_meta,
    }


def collect_candidates(events: list[EventDoc]) -> list[ReasoningCandidate]:
    candidates: list[ReasoningCandidate] = []
    for event in events:
        for doc in event.related_docs:
            candidates.extend(build_reasoning_candidates(event, doc))
    return candidates


def build_reasoning_dataset(
    *,
    input_path: Path,
    output_path: Path,
    language: str,
    limit: int,
    seed: int,
    start_id: int,
    negative_docs: int,
) -> list[dict[str, Any]]:
    data_dir, tmp = _open_mobilemem_root(input_path)
    try:
        events = [e for e in _load_events(data_dir) if e.language == language]
        rng = random.Random(seed)
        ordered = balanced_candidates(collect_candidates(events), rng)

        rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for cand in ordered:
            if limit > 0 and len(rows) >= limit:
                break
            key = candidate_key(cand)
            if key in seen_keys:
                continue
            try:
                row = make_sample(
                    cand,
                    events,
                    sample_id=start_id + len(rows),
                    negative_docs=negative_docs,
                )
            except ValueError:
                continue
            if not row:
                continue
            seen_keys.add(key)
            rows.append(row)
    finally:
        if tmp is not None:
            tmp.cleanup()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


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
    parser.add_argument("--start-id", type=int, default=320000)
    parser.add_argument("--negative-docs", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to data/rgb/<language>_mobilemem_reasoning.json.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = ROOT / "data" / "rgb" / f"{args.language}_mobilemem_reasoning.json"
    elif not output.is_absolute():
        output = ROOT / output

    rows = build_reasoning_dataset(
        input_path=args.input,
        output_path=output,
        language=args.language,
        limit=args.limit,
        seed=args.seed,
        start_id=args.start_id,
        negative_docs=args.negative_docs,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "negative_docs": args.negative_docs,
                "reasoning_types": dict(
                    sorted(
                        Counter(
                            (r.get("mobilemem_meta") or {}).get("reasoning_type")
                            for r in rows
                        ).items()
                    )
                ),
                "answer_artifact_types": dict(
                    sorted(
                        Counter(
                            (r.get("mobilemem_meta") or {}).get("answer_artifact_type")
                            for r in rows
                        ).items()
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
