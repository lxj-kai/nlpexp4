"""Build RGB-compatible RAG-hard QA data from MobileMem-Omni outputs.

The source MobileMem files describe private, synthetic yearly life traces. This
script converts each event into one RAG-dependent QA sample:

    {id, query, answer, positive, negative, positive_wrong, fakeanswer}

For MobileMem, negative documents are only semantically related but
non-supporting. Because the memories are synthetic, the default builder does
not create contradictory "counterfactual" documents: without an external truth
anchor, a model cannot know which synthetic version is true.

The generated file can be loaded by the existing experiments with:

    python -m experiments.exp1_noise_impact --language zh --subset mobilemem

Use --dry-run first to inspect candidates without calling the LLM.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mobilemem_builder")


QA_SYSTEM_ZH = """你是一个RAG评测数据构造专家。
目标：根据给定的私人生活事件证据，生成一个必须依赖这些证据才能回答的问题。

严格要求：
1. 问题必须询问事件中的具体私有细节，不能是公开常识。
2. 答案必须是证据中的原文短片段或结构化字段值；不要改写、概括、补单位、补地点或补因果。
3. 优先从群聊、应用截图、票据、金额、地址、时间、座位、聊天建议等可核查细节出题；有多个正例文档时，尽量问需要主事件+关联记录才能确定的细节。
4. 问题不能完整复制 main_event_name 或“事件名称”；可用日期、人物、场景线索定位，但不要把事件标题当作问题主体。
5. 避免问“为什么心情怎样”这类容易泛化解释的问题；优先问人物动作、具体话语、金额、时间、地点、物品、车次、座位、建议。
6. 问题和答案的语义类型必须匹配：如果问题问“具体是哪一个/哪一趟/多少钱/几点/哪句话”，答案不能只是“有一趟合适的”“还不错”“挺合适”这类泛化描述。
7. 如果证据只提供泛化描述而没有具体编号、金额、时间或原话，就把问题改成询问这个泛化描述本身，不要假装证据里有更具体的信息。
8. negative_docs 必须与主题相近，但不能支持正确答案。
9. 不要制造与正确答案矛盾的文档或答案；这些私人记忆是虚构的，没有外部真值锚点。
10. 只输出一个 JSON 对象，不要输出 Markdown，不要解释。

JSON 格式：
{
  "query": "问题",
  "answer": "正确答案，必须是证据原文短片段或结构化字段值"
}
"""


QA_SYSTEM_EN = """You are a RAG benchmark data builder.
Goal: generate one question that must be answered from the provided private life-event evidence.

Strict requirements:
1. The question must ask for private event details, not public knowledge.
2. The answer must be an exact short span or structured field value from the evidence; do not paraphrase, summarize, add units, add locations, or add causes.
3. Prefer verifiable details from chats, app screenshots, tickets, amounts, addresses, times, seats, or advice. If multiple positive docs are available, prefer questions that need the main event plus a related record.
4. The question must not copy the full main_event_name or event title. Use date, people, and scene clues instead.
5. Avoid generic emotional “why” questions; prefer concrete actions, quotes, amounts, times, places, items, train numbers, seats, or advice.
6. The question type and answer type must match: if the question asks for a specific item, train, amount, time, or quote, the answer must not be a vague description such as "a suitable one" or "quite appropriate".
7. If the evidence only provides a vague description and no concrete ID, amount, time, or quote, ask about that description itself instead of pretending the evidence contains a more specific fact.
8. negative_docs must be topically similar but must not support the correct answer.
9. Do not create documents or answers that contradict the gold answer; these private memories are synthetic and have no external truth anchor.
10. Output exactly one JSON object. No Markdown. No explanation.

JSON format:
{
  "query": "question",
  "answer": "exact short span or structured field value from the evidence"
}
"""


@dataclass
class EventDoc:
    uid: int
    language: str
    event_id: str
    event_name: str
    event_time: str
    participants: list[str]
    text: str
    related_docs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id_key(self) -> str:
        return f"{self.uid}:{self.event_id}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"skip malformed JSONL row {path}:{line_no}: {e}")
    return rows


def _open_mobilemem_root(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return a directory containing data/*.jsonl.

    The input may be the original zip, the extracted root, or the extracted data/
    directory itself.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="mobilemem_")
        tmp_path = Path(tmp.name)
        with zipfile.ZipFile(input_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.filename.startswith("data/") and (
                    info.filename.endswith(".jsonl") or info.filename.endswith(".json")
                ):
                    zf.extract(info, tmp_path)
        return tmp_path / "data", tmp

    if (input_path / "data").is_dir():
        return input_path / "data", None
    return input_path, None


def _language_from_profile(profile: dict[str, Any]) -> str:
    lang = str(profile.get("language") or profile.get("nationality") or "").lower()
    if lang in {"en", "english", "american"}:
        return "en"
    return "zh"


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _doc_from_event(event: dict[str, Any]) -> str:
    parts = [
        f"事件名称: {event.get('event_name', '')}",
        f"开始时间: {event.get('event_start_time', '')}",
        f"结束时间: {event.get('event_end_time', '')}",
        f"参与者: {', '.join(str(x) for x in event.get('participants') or [])}",
        f"描述: {event.get('description', '')}",
    ]
    extra = event.get("additional_info") or []
    if extra:
        parts.append(f"关联记录类型: {', '.join(str(x) for x in extra)}")
    return "\n".join(p for p in parts if p and not p.endswith(": "))


def _doc_from_child(child: dict[str, Any], parent: dict[str, Any]) -> str:
    parts = [
        f"长期事件: {parent.get('parent_event_name', '')}",
        f"子事件名称: {child.get('event_name', '')}",
        f"开始时间: {child.get('event_start_time', '')}",
        f"结束时间: {child.get('event_end_time', '')}",
        f"参与者: {', '.join(str(x) for x in child.get('participants') or [])}",
        f"描述: {child.get('description', '')}",
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": "))


def _index_group_chats(rows: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for row in rows:
        uid = int(row.get("uuid", -1))
        for chat in row.get("group_chats") or []:
            event_id = str(chat.get("related_event_id", ""))
            messages = []
            for msg in (chat.get("messages") or [])[:16]:
                sender = msg.get("sender", "")
                text = msg.get("text", "")
                if sender and text:
                    messages.append(f"{sender}: {text}")
            if messages:
                out[(uid, event_id)] = (
                    f"群聊名称: {chat.get('group_name', '')}\n"
                    f"关联事件: {chat.get('related_event_name', '')}\n"
                    + "\n".join(messages)
                )
    return out


def _index_app_docs(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[str]]:
    out: dict[tuple[int, str], list[str]] = {}
    for row in rows:
        uid = int(row.get("uuid", -1))
        sid = str(row.get("sub_event_id", ""))
        info = row.get("info") or {}
        if not info:
            continue
        fields = [f"{k}: {v}" for k, v in info.items()]
        doc = (
            f"应用截图类型: {row.get('app_type', '')}\n"
            f"关联事件: {row.get('event_name', '')}\n"
            + "\n".join(fields)
        )
        out.setdefault((uid, sid), []).append(doc)
    return out


def _index_ticket_docs(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[str]]:
    out: dict[tuple[int, str], list[str]] = {}
    for row in rows:
        uid = int(row.get("uuid", -1))
        sid = str(row.get("sub_event_id", ""))
        payload = row.get("ticket_info") or row.get("friend_info") or row.get("money_info") or {}
        if not payload:
            payload = {k: v for k, v in row.items() if k not in {"uuid", "image_path"}}
        fields = [f"{k}: {v}" for k, v in payload.items()]
        doc = (
            f"结构化记录类型: {row.get('type', '')}\n"
            f"关联事件: {row.get('event_name', '')}\n"
            + "\n".join(fields)
        )
        out.setdefault((uid, sid), []).append(doc)
    return out


def _load_events(data_dir: Path) -> list[EventDoc]:
    annual_rows = _read_jsonl(data_dir / "stage4_annual_events.jsonl")
    sub_rows = _read_jsonl(data_dir / "stage4_5_sub_events.jsonl")
    profiles = {
        int(row.get("uuid", -1)): row
        for row in _read_jsonl(data_dir / "stage1_basic_profiles.jsonl")
    }
    chat_by_event = _index_group_chats(_read_jsonl(data_dir / "stage7_group_chats.jsonl"))
    app_by_sub = _index_app_docs(_read_jsonl(data_dir / "stage7_2_app_screenshots.jsonl"))
    ticket_by_sub = _index_ticket_docs(_read_jsonl(data_dir / "stage7_3_tickets.jsonl"))

    events: list[EventDoc] = []

    for person in annual_rows:
        uid = int(person.get("uuid", -1))
        language = _language_from_profile(profiles.get(uid) or person.get("Basic_Profile") or {})
        for event in person.get("Events") or []:
            event_id = str(event.get("event_id", ""))
            related = []
            related.extend(app_by_sub.get((uid, event_id), []))
            related.extend(ticket_by_sub.get((uid, event_id), []))
            chat = chat_by_event.get((uid, event_id))
            if chat:
                related.append(chat)
            events.append(
                EventDoc(
                    uid=uid,
                    language=language,
                    event_id=event_id,
                    event_name=str(event.get("event_name", "")),
                    event_time=str(event.get("event_start_time", "")),
                    participants=[str(x) for x in event.get("participants") or []],
                    text=_doc_from_event(event),
                    related_docs=related,
                    metadata={
                        "source": "stage4_annual_events",
                        "additional_info": event.get("additional_info") or [],
                    },
                )
            )

    for row in sub_rows:
        uid = int(row.get("uuid", -1))
        language = _language_from_profile(profiles.get(uid) or {})
        for parent in row.get("sub_events") or []:
            for child in parent.get("children") or []:
                sid = str(child.get("sub_event_id", ""))
                related = []
                parent_id = str(parent.get("parent_event_id", ""))
                related.extend(app_by_sub.get((uid, sid), []))
                related.extend(ticket_by_sub.get((uid, sid), []))
                for chat_event_id in (sid, parent_id):
                    chat = chat_by_event.get((uid, chat_event_id))
                    if chat and chat not in related:
                        related.append(chat)
                events.append(
                    EventDoc(
                        uid=uid,
                        language=language,
                        event_id=sid,
                        event_name=str(child.get("event_name", "")),
                        event_time=str(child.get("event_start_time", "")),
                        participants=[str(x) for x in child.get("participants") or []],
                        text=_doc_from_child(child, parent),
                        related_docs=related,
                        metadata={
                            "source": "stage4_5_sub_events",
                            "parent_event_id": parent_id,
                            "parent_event_name": parent.get("parent_event_name", ""),
                            "is_intro": bool(child.get("is_intro")),
                        },
                    )
                )

    events = [e for e in events if len(e.text) >= 120]
    return events


def _similarity_key(e: EventDoc) -> set[str]:
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]+", e.text.lower()))
    tokens.update(p.lower() for p in e.participants)
    return tokens


def _choose_negative_docs(event: EventDoc, events: list[EventDoc], rng: random.Random, k: int) -> list[str]:
    target = _similarity_key(event)
    candidates: list[tuple[int, EventDoc]] = []
    for other in events:
        if other.id_key == event.id_key or other.language != event.language:
            continue
        overlap = len(target & _similarity_key(other))
        # Prefer same user and nearby themes, but allow other events as distractors.
        if other.uid == event.uid:
            overlap += 3
        candidates.append((overlap, other))
    candidates.sort(key=lambda x: x[0], reverse=True)

    docs: list[str] = []
    for _, cand in candidates[: max(20, k * 4)]:
        if cand.text not in docs:
            docs.append(cand.text)
        if len(docs) >= k:
            break
    if len(docs) < k:
        pool = [e for e in events if e.id_key != event.id_key and e.language == event.language]
        rng.shuffle(pool)
        for cand in pool:
            if cand.text not in docs:
                docs.append(cand.text)
            if len(docs) >= k:
                break
    return docs[:k]


def _build_user_prompt(event: EventDoc, negatives: list[str], *, max_positive_docs: int) -> str:
    positive_docs = _source_positive_docs(event, max_positive_docs)
    payload = {
        "main_event_id": event.id_key,
        "main_event_name": event.event_name,
        "main_event_time": event.event_time,
        "main_event_evidence": [_clip(d, 1800) for d in positive_docs],
        "candidate_negative_docs": [_clip(d, 1000) for d in negatives],
    }
    label = "输入材料" if event.language == "zh" else "Input material"
    return f"{label}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def _source_positive_docs(event: EventDoc, max_positive_docs: int) -> list[str]:
    return [event.text] + event.related_docs[: max(0, max_positive_docs - 1)]


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _norm_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[\s，。、；：？！,\.;:?!\"'`“”‘’（）()\[\]【】《》<>—\-]+", "", text)
    return text


def _contains_norm(haystack: str, needle: str) -> bool:
    n = _norm_for_match(needle)
    return bool(n) and n in _norm_for_match(haystack)


def _filter_answer_leaking_negatives(
    negatives: list[str],
    answers: list[str],
    *,
    limit: int,
) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for doc in negatives:
        doc = str(doc)
        doc_norm = _norm_for_match(doc)
        if not doc_norm or doc_norm in seen:
            continue
        if any(_contains_norm(doc, answer) for answer in answers):
            continue
        clean.append(doc)
        seen.add(doc_norm)
        if len(clean) >= limit:
            break
    return clean


def _validate_llm_sample(
    *,
    query: str,
    answer: str,
    event: EventDoc,
    source_positive_docs: list[str],
) -> None:
    positive_text = "\n".join(source_positive_docs)

    if not _contains_norm(positive_text, answer):
        raise ValueError("answer is not an exact supported span/value in positive evidence")

    event_name = _norm_for_match(event.event_name)
    if len(event_name) >= 10 and event_name in _norm_for_match(query):
        raise ValueError("query copies the full event title")


def _normalize_sample(
    obj: dict[str, Any],
    *,
    sample_id: int,
    event: EventDoc,
    source_positive_docs: list[str],
    fallback_negatives: list[str],
    negative_docs: int,
) -> dict[str, Any]:
    query = str(obj.get("query", "")).strip()
    answer = str(obj.get("answer", "")).strip()

    if not query or not answer:
        raise ValueError("missing query or answer")
    _validate_llm_sample(
        query=query,
        answer=answer,
        event=event,
        source_positive_docs=source_positive_docs,
    )
    negatives = _filter_answer_leaking_negatives(
        fallback_negatives,
        [answer],
        limit=negative_docs,
    )
    if len(negatives) < negative_docs:
        raise ValueError(
            f"only {len(negatives)}/{negative_docs} negatives remain after answer-leak filtering"
        )

    return {
        "id": sample_id,
        "query": query,
        "answer": [answer],
        "positive": source_positive_docs,
        "negative": negatives,
        "positive_wrong": [],
        "fakeanswer": "",
        "mobilemem_meta": {
            "uid": event.uid,
            "event_id": event.event_id,
            "event_name": event.event_name,
            "event_time": event.event_time,
            "noise_policy": "semantic_only_non_contradictory",
            "note": (
                "Synthetic private memories use only related non-supporting noise; "
                "contradictory counterfactual docs are excluded."
            ),
            **event.metadata,
        },
    }


def _heuristic_sample(
    *,
    sample_id: int,
    event: EventDoc,
    negatives: list[str],
) -> dict[str, Any]:
    """Fallback for --no-llm smoke generation.

    This is not as strong as LLM-generated QA, but useful for inspecting the
    pipeline and producing a few deterministic samples.
    """
    if event.language == "zh":
        query = f"在私人记忆事件“{event.event_name}”中，事件发生时间、参与者和核心经过分别是什么？"
        answer = (
            f"发生时间是{event.event_time}；参与者包括"
            f"{'、'.join(event.participants) if event.participants else '无明确参与者'}；"
            f"核心经过见该事件描述。"
        )
        fake = answer.replace(event.event_time[:10], "2025-12-31") if event.event_time else "发生时间被错误记为2025-12-31。"
        wrong_doc = event.text.replace(event.event_time[:10], "2025-12-31") if event.event_time else event.text
    else:
        query = f"In the private memory event \"{event.event_name}\", what were the time, participants, and main situation?"
        answer = (
            f"It happened at {event.event_time}; participants included "
            f"{', '.join(event.participants) if event.participants else 'no explicit participants'}; "
            "the main situation is described in the event evidence."
        )
        fake = answer.replace(event.event_time[:10], "2025-12-31") if event.event_time else "The time is wrongly stated as 2025-12-31."
        wrong_doc = event.text.replace(event.event_time[:10], "2025-12-31") if event.event_time else event.text

    return {
        "id": sample_id,
        "query": query,
        "answer": [answer],
        "positive": [event.text] + event.related_docs[:2],
        "negative": negatives[:5],
        "positive_wrong": [wrong_doc],
        "fakeanswer": fake,
        "mobilemem_meta": {
            "uid": event.uid,
            "event_id": event.event_id,
            "event_name": event.event_name,
            "event_time": event.event_time,
            **event.metadata,
        },
    }


def build_dataset(
    *,
    input_path: Path,
    output_path: Path,
    language: str,
    limit: int,
    seed: int,
    dry_run: bool,
    no_llm: bool,
    max_positive_docs: int,
    negative_docs: int,
    start_id: int,
) -> list[dict[str, Any]]:
    data_dir, tmp = _open_mobilemem_root(input_path)
    try:
        events = _load_events(data_dir)
    finally:
        # Keep tmp alive until all files are read.
        pass

    events = [e for e in events if e.language == language]
    rng = random.Random(seed)
    rng.shuffle(events)

    logger.info(f"loaded {len(events)} candidate events for language={language}")
    selected = events

    if dry_run:
        preview = selected[:10] if limit <= 0 else selected[:limit]
        for i, event in enumerate(preview, start=1):
            negatives = _choose_negative_docs(event, events, rng, negative_docs)
            print(f"\n[{i}] {event.id_key} {event.event_time} {event.event_name}")
            print(_clip(event.text, 900))
            if event.related_docs:
                print("related:", _clip(event.related_docs[0], 500))
            print("negative:", _clip(negatives[0], 500) if negatives else "")
        if tmp is not None:
            tmp.cleanup()
        return []

    if no_llm:
        llm = None
    else:
        from src.llm_client import LLMClient

        llm = LLMClient()
    samples: list[dict[str, Any]] = []
    failures = 0
    for idx, event in enumerate(selected, start=0):
        if limit > 0 and len(samples) >= limit:
            break
        sample_id = start_id + len(samples)
        negatives = _choose_negative_docs(
            event,
            events,
            rng,
            max(negative_docs * 8, negative_docs + 20),
        )
        try:
            source_positive_docs = _source_positive_docs(event, max_positive_docs)
            if no_llm:
                sample = _heuristic_sample(
                    sample_id=sample_id,
                    event=event,
                    negatives=negatives[:negative_docs],
                )
            else:
                assert llm is not None
                system = QA_SYSTEM_ZH if language == "zh" else QA_SYSTEM_EN
                base_user = _build_user_prompt(
                    event, negatives[:negative_docs], max_positive_docs=max_positive_docs
                )
                last_error: Exception | None = None
                sample = None
                for attempt in range(3):
                    reminder = ""
                    if attempt:
                        reminder = (
                            "\n\n上一次输出未通过校验。请重新生成：答案必须是正例证据中的原文短片段/"
                            "结构化字段值；不要复制完整事件标题；不要制造与答案矛盾的文档。"
                        )
                    content = llm.generate(
                        system,
                        base_user + reminder,
                        temperature=0.2 + 0.1 * attempt,
                        max_tokens=1600,
                    )
                    try:
                        obj = _extract_json_object(content)
                        sample = _normalize_sample(
                            obj,
                            sample_id=sample_id,
                            event=event,
                            source_positive_docs=source_positive_docs,
                            fallback_negatives=negatives,
                            negative_docs=negative_docs,
                        )
                        break
                    except Exception as e:
                        last_error = e
                if sample is None:
                    raise last_error or ValueError("LLM sample validation failed")
            samples.append(sample)
            if len(samples) % 10 == 0:
                total = len(selected) if limit <= 0 else limit
                logger.info(f"generated {len(samples)}/{total} samples")
        except Exception as e:
            failures += 1
            logger.warning(f"failed event {event.id_key} {event.event_name}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if tmp is not None:
        tmp.cleanup()

    logger.info(f"wrote {len(samples)} samples -> {output_path}; failures={failures}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT.parent / "data" / "0418(1).zip",
        help="MobileMem zip, extracted root, or data/ directory.",
    )
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--limit", type=int, default=50, help="Number of events to convert; 0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Generate deterministic rough samples without API calls.",
    )
    parser.add_argument("--max-positive-docs", type=int, default=3)
    parser.add_argument("--negative-docs", type=int, default=5)
    parser.add_argument("--start-id", type=int, default=300000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to data/rgb/<language>_mobilemem.json.",
    )
    parser.epilog = (
        "Note: MobileMem samples intentionally leave positive_wrong/fakeanswer empty. "
        "Use semantic noise for the main RAG robustness experiment; contradictory "
        "counterfactual documents are not valid ordinary noise for synthetic memories."
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = ROOT / "data" / "rgb" / f"{args.language}_mobilemem.json"
    elif not output.is_absolute():
        output = ROOT / output

    build_dataset(
        input_path=args.input,
        output_path=output,
        language=args.language,
        limit=args.limit,
        seed=args.seed,
        dry_run=args.dry_run,
        no_llm=args.no_llm,
        max_positive_docs=args.max_positive_docs,
        negative_docs=args.negative_docs,
        start_id=args.start_id,
    )


if __name__ == "__main__":
    main()
