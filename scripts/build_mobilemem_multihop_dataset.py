"""Build 3-hop bridge-entity RAG samples from MobileMem memories.

Design (rigorous + hard, see chat design review):

    Q anchors a unique memory A  ->  the single co-participant B (bridge)
    ->  among B's <kind> records inside a time window W, the one with the
        highest <metric>  ->  answer = that record's ENTITY name.

Invariants enforced per sample (else dropped):
  - bridge non-co-occurrence: anchor kind != target kind, so the anchor
    descriptor never appears in the answer doc and vice-versa (no shortcut).
  - unique answer: the in-window B set has a single strict max.
  - window_needed: there exists a B record OUTSIDE W with a HIGHER value
    (a model that ignores the window is trapped).
  - bridge_needed: there exists an in-window record by ANOTHER person with a
    HIGHER value (a model that ignores the bridge is trapped).
  - leak=0: no negative is itself a valid (B & in-window & kind) answer.
  - answer entity is unique across all docs in the sample.

Answers are entities (item / title / book name), never raw numbers, so the
clean task is easy and grading is robust; difficulty lives entirely in
whether noise breaks the chain.

Output stays RGB-compatible:
    {id, query, answer, positive, negative, positive_wrong, fakeanswer}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_graph_hard_dataset import (  # noqa: E402
    Artifact,
    collect_artifacts,
    parse_number,
    parse_reading_minutes,
)
from scripts.build_mobilemem_rag_dataset import (  # noqa: E402
    _load_events,
    _open_mobilemem_root,
)

OWNER_NAMES = {"我", "本人", "自己"}  # generic; per-uid owner added dynamically

# kind -> (aggregate phrase, record label, answer label)
TARGET_KINDS: dict[str, tuple[str, str, str]] = {
    "shopping": ("实际价格最高", "购物记录", "商品名称"),
    "video": ("点赞数最高", "刷到/收藏的视频记录", "视频标题"),
    "book": ("阅读时长最长", "读书记录", "书名"),
}
# kind -> short action phrase used to describe the anchor memory
ANCHOR_ACTION: dict[str, str] = {
    "book": "读过的一本书",
    "music": "听过的一首歌",
    "video": "刷到并收藏的一条视频",
    "shopping": "买过的一件商品",
    "ticket": "买过的一张行程车票",
}


def parse_time(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def art_value(a: Artifact) -> float | None:
    if a.kind == "shopping":
        return parse_number(a.fields.get("price", ""))
    if a.kind == "video":
        return parse_number(a.fields.get("like_count", ""))
    if a.kind == "book":
        return parse_reading_minutes(a.fields.get("reading_time", ""))
    return None


def art_entity(a: Artifact) -> str:
    if a.kind == "shopping":
        return (a.fields.get("item_name") or "").strip()
    if a.kind in ("video", "book"):
        return (a.fields.get("title") or "").strip()
    if a.kind == "music":
        return (a.fields.get("song") or "").strip()
    if a.kind == "ticket":
        return (a.fields.get("train_number") or "").strip()
    return ""


def load_owners(data_dir: Path) -> dict[int, str]:
    """uid -> owner (self) name, so it can be excluded from participants."""
    path = data_dir / "stage3_9_social_graph.jsonl"
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name = (r.get("Basic_Profile") or {}).get("name")
        if r.get("uuid") is not None and name:
            out[int(r["uuid"])] = str(name)
    return out


def load_participants_map(data_dir: Path, owners: dict[int, str]) -> dict[tuple, list[str]]:
    """(uid, event_id, app_type, entity) -> [non-owner participants]."""
    path = data_dir / "stage7_2_app_screenshots.jsonl"
    out: dict[tuple, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        info = r.get("info") or {}
        ent = (
            info.get("item_name")
            or info.get("title")
            or info.get("song")
            or info.get("train_number")
            or ""
        ).strip()
        uid = int(r.get("uuid"))
        owner = owners.get(uid)
        people = [
            p for p in (r.get("participants") or [])
            if p not in OWNER_NAMES and p != owner
        ]
        out[(uid, str(r.get("event_id")), r.get("app_type"), ent)] = people
    return out


@dataclass
class Mem:
    art: Artifact
    people: list[str]
    value: float
    entity: str
    time: datetime
    uid: int

    @property
    def kind(self) -> str:
        return self.art.kind

    @property
    def doc(self) -> str:
        base = self.art.compact_doc
        if self.people:
            base += "\n同行好友: " + "、".join(self.people)
        return base


def build_mems(data_dir: Path, owners: dict[int, str]) -> list[Mem]:
    events = _load_events(data_dir)
    arts = collect_artifacts(events)
    pmap = load_participants_map(data_dir, owners)
    mems: list[Mem] = []
    for a in arts:
        ent = art_entity(a)
        val = art_value(a)
        t = parse_time(a.event.event_time)
        people = pmap.get((int(a.event.uid), str(a.event.event_id), a.kind, ent), [])
        mems.append(
            Mem(art=a, people=people, value=val if val is not None else float("nan"),
                entity=ent, time=t, uid=a.event.uid)
        )
    return mems


def calendar_windows(year: int = 2025) -> list[tuple[str, datetime, datetime]]:
    """Natural windows (quarters + months) so the window is independent of the
    answer set — never a post-hoc span between the gold items."""
    wins: list[tuple[str, datetime, datetime]] = []
    qmonths = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
    for q, (m0, m1) in qmonths.items():
        t1 = datetime(year, m0, 1, 0, 0)
        t2 = datetime(year, m1, 28, 23, 59) if m1 == 2 else datetime(
            year, m1, 30 if m1 in (4, 6, 9, 11) else 31, 23, 59)
        wins.append((f"{year}年第{q}季度（{t1:%Y-%m-%d} 至 {t2:%Y-%m-%d}）", t1, t2))
    for m in range(1, 13):
        t1 = datetime(year, m, 1, 0, 0)
        last = 30 if m in (4, 6, 9, 11) else (28 if m == 2 else 31)
        t2 = datetime(year, m, last, 23, 59)
        wins.append((f"{year}年{m}月（{t1:%Y-%m-%d} 至 {t2:%Y-%m-%d}）", t1, t2))
    return wins


def build_query(anchor: Mem, bridge: str, target_kind: str, win_label: str) -> str:
    agg, rec_label, ans_label = TARGET_KINDS[target_kind]
    action = ANCHOR_ACTION.get(anchor.art.kind, "记录过的一件事")
    higher = agg.replace("最高", "更高").replace("最长", "更长")
    return (
        f"在我的 MobileMem 私人记忆里，有一条记忆是我{action}：「{anchor.entity}」，"
        f"那条记忆里和我一起的好友只有一位（记为 TA）。"
        f"请在 {win_label} 内，找出我和 TA 共同参与的{rec_label}中{agg}的那一条，"
        f"只回答它的{ans_label}。"
        f"注意：不要计入 TA 以外的人的记录，也不要计入不在该时间段内的记录，"
        f"即使它们主题相近或{higher}。"
    )


def doc_set(mems: list[Mem]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in mems:
        d = m.doc
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def make_samples(
    mems: list[Mem],
    rng: random.Random,
    *,
    uid: int,
    target_kinds: list[str],
    max_negatives: int,
    per_bridge: int,
    windows: list[tuple[str, datetime, datetime]],
    min_in: int,
    trap_mode: str = "dual",
) -> list[dict[str, Any]]:
    mems = [m for m in mems if m.uid == uid and m.time is not None]
    valid = [m for m in mems if m.value == m.value and m.entity]  # not NaN, has entity

    # person -> kind -> [Mem]
    by_person: dict[str, dict[str, list[Mem]]] = defaultdict(lambda: defaultdict(list))
    for m in valid:
        for p in m.people:
            by_person[p][m.kind].append(m)
    # kind -> [Mem] (any participants) for bridge-trap search
    by_kind: dict[str, list[Mem]] = defaultdict(list)
    for m in valid:
        by_kind[m.kind].append(m)
    # single-participant mems per person for anchors
    single: dict[str, list[Mem]] = defaultdict(list)
    for m in valid:
        if len(m.people) == 1:
            single[m.people[0]].append(m)

    samples: list[dict[str, Any]] = []
    used_keys: set[tuple] = set()
    bridges = sorted(by_person, key=lambda p: -sum(len(v) for v in by_person[p].values()))
    for bridge in bridges:
        made = 0
        for tkind in target_kinds:
            if made >= per_bridge:
                break
            bk = by_person[bridge][tkind]
            if len(bk) < 2:
                continue
            # candidate windows: natural calendar periods first (preferred),
            # then padded runs of B's items (boundaries rounded to ±3 days so
            # they are natural dates that never coincide with an item timestamp).
            bk_sorted = sorted(bk, key=lambda m: m.time)
            run_windows: list[tuple[str, datetime, datetime]] = []
            for size in (2, 3):
                for i in range(0, len(bk_sorted) - size + 1):
                    r = bk_sorted[i : i + size]
                    rt1 = (r[0].time - timedelta(days=3)).replace(hour=0, minute=0)
                    rt2 = (r[-1].time + timedelta(days=3)).replace(hour=23, minute=59)
                    run_windows.append((f"{rt1:%Y-%m-%d} 至 {rt2:%Y-%m-%d}", rt1, rt2))
            cand_windows = list(windows) + run_windows
            for win_label, t1, t2 in cand_windows:
                if made >= per_bridge:
                    break
                in_set = [m for m in bk if t1 <= m.time <= t2]
                if len(in_set) < min_in:
                    continue
                for _once in (0,):  # preserve original body indentation
                    vals = sorted((m.value for m in in_set), reverse=True)
                    if len(vals) < 2 or vals[0] == vals[1]:
                        continue  # need strict unique max
                    target = max(in_set, key=lambda m: m.value)
                    tv = target.value
                    # window_needed: B record outside W with higher value
                    win_trap = [m for m in bk if not (t1 <= m.time <= t2) and m.value > tv]
                    # bridge_needed: in-window record by another person, higher value
                    brg_trap = [
                        m for m in by_kind[tkind]
                        if t1 <= m.time <= t2 and bridge not in m.people and m.value > tv
                    ]
                    if trap_mode == "dual":
                        if not win_trap or not brg_trap:
                            continue
                    else:  # single: at least one adversarial higher-value trap
                        if not win_trap and not brg_trap:
                            continue
                    gold = target.entity
                    if (bridge, tkind, gold) in used_keys:
                        continue
                    # anchor: single-participant mem of B, kind != target, on an
                    # event disjoint from every candidate AND every trap (so the
                    # anchor can never double as evidence / a shortcut)
                    used_events = (
                        {m.art.event.event_id for m in in_set}
                        | {m.art.event.event_id for m in win_trap[:2]}
                        | {m.art.event.event_id for m in brg_trap[:2]}
                    )
                    anc_choices = [
                        m for m in single[bridge]
                        if m.kind != tkind and m.art.event.event_id not in used_events
                    ]
                    if not anc_choices:
                        continue
                    anchor = max(anc_choices, key=lambda m: len(m.entity))
                    # assemble positives: anchor + in-window B set
                    positives = [anchor] + in_set
                    # --- noise ordered so the hardest designed traps come FIRST;
                    # record.negative order == injection priority, so even low
                    # noise levels hit the real traps (not luck). Traps sorted by
                    # closeness to gold (competitive) so they are hard to tell apart.
                    brg_sorted = sorted(brg_trap, key=lambda m: m.value)
                    win_sorted = sorted(win_trap, key=lambda m: m.value)
                    counter_anchor = [
                        m for m in valid
                        if len(m.people) == 1 and m.people[0] != bridge
                        and m.kind == anchor.kind
                        and m.art.event.event_id not in used_events
                    ]
                    rng.shuffle(counter_anchor)
                    priority: list[Mem] = []
                    priority += brg_sorted[:2]      # in-window, other person, higher
                    priority += win_sorted[:2]      # out-window, bridge, higher
                    priority += counter_anchor[:2]  # same anchor topic, different person
                    other_kind_B = [
                        m for m in valid
                        if bridge in m.people and m.kind not in (tkind, anchor.kind)
                    ]
                    fillers = [
                        m for m in by_kind[tkind]
                        if m not in in_set and m not in win_trap and m not in brg_trap
                    ]
                    rng.shuffle(fillers)
                    negs = priority + other_kind_B[:2] + fillers
                    # dedupe, cap, and drop any that equals an in-set/anchor doc
                    pos_docs = doc_set(positives)
                    pos_doc_set = set(pos_docs)
                    neg_docs: list[str] = []
                    seen = set(pos_doc_set)
                    for m in negs:
                        d = m.doc
                        if d in seen:
                            continue
                        seen.add(d)
                        neg_docs.append(d)
                        if len(neg_docs) >= max_negatives:
                            break
                    # audits
                    leak = any(gold and gold in d for d in neg_docs)
                    shortcut = (gold in anchor.doc) or (anchor.entity in target.doc)
                    if leak or shortcut:
                        continue
                    sample = {
                        "query": build_query(anchor, bridge, tkind, win_label),
                        "answer": [gold],
                        "positive": pos_docs,
                        "negative": neg_docs,
                        "positive_wrong": [],
                        "fakeanswer": "",
                        "mobilemem_meta": {
                            "uid": uid,
                            "question_mode": "multihop_bridge",
                            "hardness": "bridge_window_max",
                            "bridge_person": bridge,
                            "anchor_kind": anchor.kind,
                            "anchor_entity": anchor.entity,
                            "target_kind": tkind,
                            "aggregate": "max",
                            "window_label": win_label,
                            "window": [t1.strftime("%Y-%m-%d %H:%M"), t2.strftime("%Y-%m-%d %H:%M")],
                            "gold_entity": gold,
                            "gold_value": tv,
                            "in_window_count": len(in_set),
                            "window_trap_value": min((m.value for m in win_trap), default=None),
                            "bridge_trap_value": min((m.value for m in brg_trap), default=None),
                            "n_counter_anchor": len(counter_anchor),
                            "priority_traps": len(priority),
                            "n_positive": len(pos_docs),
                            "n_negative": len(neg_docs),
                        },
                    }
                    samples.append(sample)
                    used_keys.add((bridge, tkind, gold))
                    made += 1
                    break  # stop this window; continue to next window/kind
    return samples


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter
    return {
        "rows": len(rows),
        "bridges": dict(Counter(r["mobilemem_meta"]["bridge_person"] for r in rows)),
        "target_kinds": dict(Counter(r["mobilemem_meta"]["target_kind"] for r in rows)),
        "avg_negatives": round(sum(r["mobilemem_meta"]["n_negative"] for r in rows) / max(1, len(rows)), 1),
        "avg_in_window": round(sum(r["mobilemem_meta"]["in_window_count"] for r in rows) / max(1, len(rows)), 1),
    }


def parse_uids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "20.zip")
    parser.add_argument("--uids", type=str, default="0-9",
                        help="persona uids to batch over, e.g. '0-9' or '0,2,5'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-id", type=int, default=350000)
    parser.add_argument("--max-negatives", type=int, default=40)
    parser.add_argument("--per-bridge", type=int, default=3)
    parser.add_argument("--min-in-window", type=int, default=3,
                        help="min B records inside the window (richer 'max').")
    parser.add_argument("--trap-mode", choices=("dual", "single"), default="dual",
                        help="dual=window AND bridge higher-value traps; single=at least one.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--target-kind", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "rgb" / "zh_mobilemem_multihop.json")
    args = parser.parse_args()

    target_kinds = args.target_kind or list(TARGET_KINDS)
    uids = parse_uids(args.uids)
    data_dir, _tmp = _open_mobilemem_root(args.input)
    owners = load_owners(data_dir)
    mems = build_mems(data_dir, owners)

    samples: list[dict[str, Any]] = []
    for uid in uids:
        rng = random.Random(args.seed + uid)
        samples += make_samples(
            mems, rng, uid=uid, target_kinds=target_kinds,
            max_negatives=args.max_negatives, per_bridge=args.per_bridge,
            windows=calendar_windows(), min_in=args.min_in_window,
            trap_mode=args.trap_mode,
        )
    if args.limit:
        samples = samples[: args.limit]
    for i, s in enumerate(samples):
        s["id"] = args.start_id + i

    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = audit(samples)
    summary["uids"] = uids
    out.with_suffix(".audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(out), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
