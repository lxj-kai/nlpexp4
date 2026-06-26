"""Build MobileMem-Hard QA data with anchor+artifact evidence chains.

MobileMem-Hard is stricter than the basic MobileMem QA set:

- each question has exactly two positive documents;
- the first positive is the narrative anchor used to identify the private event;
- the second positive is an artifact record, such as chat, ticket, money,
  shopping, music, video, book, or social post, that contains the answer;
- the answer must not appear in the narrative anchor;
- negatives are close, non-contradictory, non-supporting documents.

The resulting data is still RGB-compatible:

    {id, query, answer, positive, negative, positive_wrong, fakeanswer}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_rag_dataset import (  # noqa: E402
    EventDoc,
    _contains_norm,
    _load_events,
    _norm_for_match,
    _open_mobilemem_root,
    _similarity_key,
)


FIELD_QUESTIONS: dict[str, dict[str, str]] = {
    "shopping": {
        "item_name": "购物记录里的商品名称是什么？",
        "shop_name": "购物记录里的店铺名称是什么？",
        "price": "购物记录里的实际价格是多少？",
        "order_time": "购物记录里的下单时间是什么？",
    },
    "money": {
        "recipient_name": "转账记录里的收款方是谁？",
        "amount": "转账记录里的金额是多少？",
        "transfer_time": "转账记录里的转账时间是什么？",
        "transaction_id": "转账记录里的交易单号是什么？",
    },
    "ticket": {
        "train_number": "票据记录里的车次是什么？",
        "seat_number": "票据记录里的座位号是什么？",
        "departure_station": "票据记录里的出发站是什么？",
        "arrival_station": "票据记录里的到达站是什么？",
    },
    "music": {
        "song": "音乐记录里的歌曲名是什么？",
        "artist": "音乐记录里的歌手是谁？",
        "current_time": "音乐记录里的当前播放时间是多少？",
        "playlist": "音乐记录里的歌单名称是什么？",
        "comment": "音乐记录里的评论内容是什么？",
    },
    "book": {
        "title": "读书记录里的书名是什么？",
        "author": "读书记录里的作者是谁？",
        "progress": "读书记录里的阅读进度是什么？",
        "reading_time": "读书记录里的阅读时长是多少？",
    },
    "video": {
        "title": "视频记录里的标题是什么？",
        "uploader": "视频记录里的上传者是谁？",
        "duration": "视频记录里的时长是多少？",
        "view_count": "视频记录里的播放量是多少？",
        "danmaku_count": "视频记录里的弹幕数是多少？",
        "like_count": "视频记录里的点赞数是多少？",
    },
    "friend": {
        "post_text": "朋友圈记录里的正文是什么？",
        "post_time": "朋友圈记录里的发布时间是什么？",
    },
}

ARTIFACT_LABEL = {
    "shopping": "购物截图",
    "money": "转账记录",
    "ticket": "票据记录",
    "music": "音乐截图",
    "book": "读书截图",
    "video": "视频截图",
    "friend": "朋友圈记录",
    "chat": "群聊记录",
}

ORDINAL_ZH = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


@dataclass(frozen=True)
class Candidate:
    event: EventDoc
    answer_doc: str
    answer: str
    question_tail: str
    artifact_type: str
    answer_field: str


def parse_artifact(doc: str) -> tuple[str, dict[str, str], list[tuple[str, str]]]:
    lines = [line.strip() for line in doc.splitlines() if line.strip()]
    if not lines:
        return "unknown", {}, []

    kind = "unknown"
    if lines[0].startswith("应用截图类型:"):
        kind = lines[0].split(":", 1)[1].strip()
    elif lines[0].startswith("结构化记录类型:"):
        raw = lines[0].split(":", 1)[1].strip()
        kind = raw if raw in {"money", "ticket", "friend"} else f"struct_{raw}"
    elif lines[0].startswith("群聊名称:"):
        kind = "chat"

    fields: dict[str, str] = {}
    messages: list[tuple[str, str]] = []
    for line in lines[1:]:
        if kind == "chat" and ": " in line:
            sender, text = line.split(": ", 1)
            if sender and text:
                messages.append((sender.strip(), text.strip()))
            continue
        if ": " in line:
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip()
    return kind, fields, messages


def description_text(event: EventDoc) -> str:
    marker = "描述: "
    if marker in event.text:
        return event.text.split(marker, 1)[1].strip()
    return event.text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])", re.sub(r"\s+", " ", text).strip())
    return [p.strip(" ，。、；;") for p in parts if p.strip(" ，。、；;")]


def anchor_clue(event: EventDoc, answer: str) -> str | None:
    for sent in split_sentences(description_text(event)):
        if _contains_norm(sent, answer):
            continue
        if 18 <= len(sent) <= 86:
            return sent
    text = description_text(event)
    if _contains_norm(text[:90], answer):
        return None
    return text[:70].strip(" ，。、；;") if len(text) >= 25 else None


def valid_answer(answer: object, event: EventDoc, answer_doc: str) -> str | None:
    text = str(answer).strip()
    if not text:
        return None
    if len(text) > 90:
        return None
    norm = _norm_for_match(text)
    if len(norm) < 2:
        return None
    if text in {"已完成", "收藏", "点赞", "在线支付", "微信读书"}:
        return None
    if _contains_norm(event.text, text):
        return None
    if not _contains_norm(answer_doc, text):
        return None
    return text


def build_field_candidates(event: EventDoc, doc: str) -> list[Candidate]:
    kind, fields, messages = parse_artifact(doc)
    out: list[Candidate] = []

    if kind == "chat":
        sender_counts: dict[str, int] = {}
        for sender, text in messages[:12]:
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
            answer = valid_answer(text, event, doc)
            if not answer:
                continue
            if len(answer) < 8 or len(answer) > 70:
                continue
            ordinal = ORDINAL_ZH[sender_counts[sender] - 1] if sender_counts[sender] <= len(ORDINAL_ZH) else str(sender_counts[sender])
            out.append(
                Candidate(
                    event=event,
                    answer_doc=doc,
                    answer=answer,
                    question_tail=f"相关群聊里，{sender}发出的第{ordinal}条消息是什么？",
                    artifact_type="chat",
                    answer_field=f"{sender}_{sender_counts[sender]}",
                )
            )
            if len(out) >= 2:
                break
        return out

    questions = FIELD_QUESTIONS.get(kind, {})
    for field, question_tail in questions.items():
        if field not in fields:
            continue
        answer = valid_answer(fields[field], event, doc)
        if not answer:
            continue
        out.append(
            Candidate(
                event=event,
                answer_doc=doc,
                answer=answer,
                question_tail=question_tail,
                artifact_type=kind,
                answer_field=field,
            )
        )
    return out


def candidate_key(c: Candidate) -> tuple[str, str, str]:
    return c.event.id_key, c.artifact_type, c.answer_field


def build_query(candidate: Candidate) -> str | None:
    clue = anchor_clue(candidate.event, candidate.answer)
    if not clue:
        return None
    label = ARTIFACT_LABEL.get(candidate.artifact_type, "关联记录")
    query = (
        f"在私人记忆中，先根据“{clue}”定位对应事件；"
        f"再查看同一事件的{label}，{candidate.question_tail}"
    )
    if _contains_norm(query, candidate.answer):
        return None
    event_name = _norm_for_match(candidate.event.event_name)
    if len(event_name) >= 10 and event_name in _norm_for_match(query):
        return None
    return query


def parent_id(event: EventDoc) -> str:
    return str(event.metadata.get("parent_event_id") or event.event_id).split("_", 1)[0]


def all_event_docs(event: EventDoc) -> list[tuple[str, str]]:
    docs = [(event.text, "event")]
    for doc in event.related_docs:
        kind, _, _ = parse_artifact(doc)
        docs.append((doc, kind))
    return docs


def choose_negatives(
    candidate: Candidate,
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


def make_sample(
    candidate: Candidate,
    events: list[EventDoc],
    *,
    sample_id: int,
    negative_docs: int,
) -> dict[str, Any] | None:
    query = build_query(candidate)
    if not query:
        return None
    negatives, negative_meta = choose_negatives(candidate, events, count=negative_docs)
    return {
        "id": sample_id,
        "query": query,
        "answer": [candidate.answer],
        "positive": [candidate.event.text, candidate.answer_doc],
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
            "hardness": "anchor_plus_artifact",
            "answer_artifact_type": candidate.artifact_type,
            "answer_field": candidate.answer_field,
            "requires_anchor_doc": True,
            "requires_artifact_doc": True,
            "negative_docs": negative_docs,
        },
        "negative_meta": negative_meta,
    }


def collect_candidates(events: list[EventDoc]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for event in events:
        for doc in event.related_docs:
            candidates.extend(build_field_candidates(event, doc))
    return candidates


def build_hard_dataset(
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
    finally:
        pass

    rng = random.Random(seed)
    event_by_id = {e.id_key: e for e in events}
    all_candidates = collect_candidates(events)
    rng.shuffle(all_candidates)

    # First pass prefers event diversity, second pass fills with additional
    # fields from already-used events if necessary.
    first_by_event: dict[str, Candidate] = {}
    extras: list[Candidate] = []
    for cand in all_candidates:
        if cand.event.id_key not in first_by_event:
            first_by_event[cand.event.id_key] = cand
        else:
            extras.append(cand)
    ordered = list(first_by_event.values()) + extras

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for cand in ordered:
        if limit > 0 and len(rows) >= limit:
            break
        key = candidate_key(cand)
        if key in seen_keys:
            continue
        if cand.event.id_key not in event_by_id:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if tmp is not None:
        tmp.cleanup()
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
    parser.add_argument("--start-id", type=int, default=310000)
    parser.add_argument("--negative-docs", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to data/rgb/<language>_mobilemem_hard.json.",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = ROOT / "data" / "rgb" / f"{args.language}_mobilemem_hard.json"
    elif not output.is_absolute():
        output = ROOT / output

    rows = build_hard_dataset(
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
                "answer_artifact_types": sorted(
                    {
                        (r.get("mobilemem_meta") or {}).get("answer_artifact_type")
                        for r in rows
                    }
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
