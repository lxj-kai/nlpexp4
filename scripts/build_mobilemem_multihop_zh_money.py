"""Chinese multi-hop bridge samples from 20.zip personas (transfer-based).

The zh personas (uid 0-9) have no shopping screenshots; their value-bearing
records are transfers in money.jsonl. Tickets (different kind) serve as the
bridge-revealing anchor so there is no shortcut / no anchor-as-candidate.

    Q anchors a unique 车票 (route) -> its single co-traveller B (bridge)
    -> among B's 转账 inside a time window W, the one with the highest 金额
    -> answer = that transfer's 事由 (description).

Same rigor as the shopping generator: unique strict max, window_trap,
bridge_trap, counter-anchor, leak=0, traps ordered first (priority injection),
competitive (closest to gold). RGB-compatible output (same meta schema) so the
existing clean filter / subset / frontend work unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_mobilemem_graph_hard_dataset import parse_number  # noqa: E402
from scripts.build_mobilemem_multihop_dataset import (  # noqa: E402
    calendar_windows,
    load_owners,
    parse_time,
)
from scripts.build_mobilemem_rag_dataset import _open_mobilemem_root  # noqa: E402

OWNER_GENERIC = {"我", "本人", "自己"}


def fmt_amount(v: float) -> str:
    return f"{v:.0f}元" if abs(v - round(v)) < 1e-6 else f"{v:.2f}元"


@dataclass
class Transfer:
    uid: int
    event_id: str
    time: datetime
    amount: float
    recipient: str
    desc: str
    people: list[str]
    lang: str = "zh"

    @property
    def doc(self) -> str:
        if self.lang == "en":
            lines = [
                "Memory evidence: MobileMem personal-life record",
                f"Transfer time: {self.time:%Y-%m-%d %H:%M}",
                "Record type: Transfer",
                f"Payee: {self.recipient}",
                f"Amount: {self.amount:.2f}",
                f"Purpose: {self.desc}",
            ]
            if self.people:
                lines.append("Companions: " + ", ".join(self.people))
            return "\n".join(lines)
        lines = [
            "记忆证据: MobileMem 私人生活记录",
            f"转账时间: {self.time:%Y-%m-%d %H:%M}",
            "记录类型: 转账记录",
            f"收款方: {self.recipient}",
            f"转账金额: {fmt_amount(self.amount)}",
            f"事由: {self.desc}",
        ]
        if self.people:
            lines.append("同行好友: " + "、".join(self.people))
        return "\n".join(lines)


@dataclass
class Ticket:
    uid: int
    event_id: str
    dep: str
    arr: str
    train: str
    people: list[str]
    lang: str = "zh"

    @property
    def label(self) -> str:
        if self.lang == "en":
            return f"from {self.dep} to {self.arr} (train {self.train})"
        return f"从{self.dep}到{self.arr}（车次{self.train}）"

    @property
    def doc(self) -> str:
        if self.lang == "en":
            lines = [
                "Memory evidence: MobileMem personal-life record",
                "Record type: Travel ticket",
                f"Departure: {self.dep}",
                f"Arrival: {self.arr}",
                f"Train no.: {self.train}",
            ]
            if self.people:
                lines.append("Companions: " + ", ".join(self.people))
            return "\n".join(lines)
        lines = [
            "记忆证据: MobileMem 私人生活记录",
            "记录类型: 行程车票",
            f"出发站: {self.dep}",
            f"到达站: {self.arr}",
            f"车次: {self.train}",
        ]
        if self.people:
            lines.append("同行好友: " + "、".join(self.people))
        return "\n".join(lines)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def load_transfers(data_dir: Path, owners: dict[int, str]) -> list[Transfer]:
    out: list[Transfer] = []
    for r in read_jsonl(data_dir / "money.jsonl"):
        uid = int(r.get("uuid"))
        mi = r.get("money_info") or {}
        t = parse_time(mi.get("transfer_time", ""))
        amt = parse_number(str(mi.get("amount", "")))
        desc = (mi.get("description") or "").strip()
        if t is None or amt is None or not desc:
            continue
        owner = owners.get(uid)
        people = [p for p in (r.get("participants") or []) if p not in OWNER_GENERIC and p != owner]
        out.append(Transfer(uid, str(r.get("sub_event_id")), t, amt,
                            (mi.get("recipient_name") or "").strip(), desc, people))
    return out


def load_tickets(data_dir: Path, owners: dict[int, str]) -> list[Ticket]:
    out: list[Ticket] = []
    for r in read_jsonl(data_dir / "ticket.jsonl"):
        uid = int(r.get("uuid"))
        ti = r.get("ticket_info") or {}
        owner = owners.get(uid)
        people = [p for p in (r.get("participants") or []) if p not in OWNER_GENERIC and p != owner]
        train = (ti.get("train_number") or "").strip()
        dep = (ti.get("departure_station") or "").strip()
        arr = (ti.get("arrival_station") or "").strip()
        if not (train and dep and arr):
            continue
        out.append(Ticket(uid, str(r.get("sub_event_id")), dep, arr, train, people))
    return out


def is_exact_bridge_transfer(t: Transfer, bridge: str) -> bool:
    return t.people == [bridge]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audit_money(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    leaks = []
    bad_positive_answer_counts = []
    for row in rows:
        answer = str((row.get("answer") or [""])[0])
        positives = row.get("positive") or []
        negatives = row.get("negative") or []
        pos_hits = sum(1 for doc in positives if answer and answer in doc)
        neg_hits = sum(1 for doc in negatives if answer and answer in doc)
        if pos_hits != 1:
            bad_positive_answer_counts.append({"id": row.get("id"), "pos_hits": pos_hits})
        if neg_hits:
            leaks.append({"id": row.get("id"), "neg_hits": neg_hits})
    metas = [row.get("mobilemem_meta") or {} for row in rows]
    return {
        "rows": len(rows),
        "bridges": dict(Counter(meta.get("bridge_person") for meta in metas)),
        "target_kinds": dict(Counter(meta.get("target_kind") for meta in metas)),
        "min_positives": min((len(row.get("positive") or []) for row in rows), default=0),
        "max_positives": max((len(row.get("positive") or []) for row in rows), default=0),
        "min_negatives": min((len(row.get("negative") or []) for row in rows), default=0),
        "max_negatives": max((len(row.get("negative") or []) for row in rows), default=0),
        "avg_negatives": round(
            sum(meta.get("n_negative", 0) for meta in metas) / max(1, len(rows)), 1
        ),
        "avg_in_window": round(
            sum(meta.get("in_window_count", 0) for meta in metas) / max(1, len(rows)), 1
        ),
        "dual_trap_rows": sum(
            1
            for meta in metas
            if meta.get("window_trap_value") is not None
            and meta.get("bridge_trap_value") is not None
        ),
        "exact_two_target_rows": sum(1 for meta in metas if meta.get("in_window_count") == 2),
        "leak_count": len(leaks),
        "bad_positive_answer_count": len(bad_positive_answer_counts),
        "leak_examples": leaks[:5],
        "bad_positive_answer_examples": bad_positive_answer_counts[:5],
    }


def make_samples(
    transfers,
    tickets,
    rng,
    *,
    uid,
    windows,
    min_in,
    max_negatives,
    per_bridge,
    trap_mode,
    exact_bridge_only,
    exact_in_window,
    min_positive_margin,
    min_positive_ratio,
    min_gold_amount,
    prefer_gold_latest,
):
    tr = [t for t in transfers if t.uid == uid]
    tk = [k for k in tickets if k.uid == uid]
    by_person: dict[str, list[Transfer]] = defaultdict(list)
    for t in tr:
        for p in t.people:
            by_person[p].append(t)
    # single-traveller tickets per person (unique bridge anchors)
    single_ticket: dict[str, list[Ticket]] = defaultdict(list)
    for k in tk:
        if len(k.people) == 1:
            single_ticket[k.people[0]].append(k)
    all_single_tickets = [k for k in tk if len(k.people) == 1]

    samples: list[dict[str, Any]] = []
    used: set[tuple] = set()
    bridges = sorted(by_person, key=lambda p: -len(by_person[p]))
    for bridge in bridges:
        if bridge not in single_ticket:
            continue  # need a ticket anchor that uniquely names the bridge
        bt = by_person[bridge]
        if len(bt) < 2:
            continue
        bt_sorted = sorted(bt, key=lambda t: t.time)
        run_windows = []
        for size in (2, 3):
            for i in range(0, len(bt_sorted) - size + 1):
                r = bt_sorted[i:i + size]
                rt1 = (r[0].time - timedelta(days=3)).replace(hour=0, minute=0)
                rt2 = (r[-1].time + timedelta(days=3)).replace(hour=23, minute=59)
                run_windows.append((f"{rt1:%Y-%m-%d} 至 {rt2:%Y-%m-%d}", rt1, rt2))
        made = 0
        for win_label, t1, t2 in list(windows) + run_windows:
            if made >= per_bridge:
                break
            in_set = [
                t
                for t in bt
                if t1 <= t.time <= t2
                and (not exact_bridge_only or is_exact_bridge_transfer(t, bridge))
            ]
            if exact_in_window > 0 and len(in_set) != exact_in_window:
                continue
            if len(in_set) < min_in:
                continue
            vals = sorted((t.amount for t in in_set), reverse=True)
            if vals[0] == vals[1]:
                continue
            target = max(in_set, key=lambda t: t.amount)
            tv = target.amount
            runner_up = vals[1]
            if tv < min_gold_amount:
                continue
            if tv - runner_up < min_positive_margin:
                continue
            if runner_up > 0 and tv / runner_up < min_positive_ratio:
                continue
            gold = fmt_amount(tv)
            if (bridge, gold) in used:
                continue
            win_trap = [
                t
                for t in bt
                if not (t1 <= t.time <= t2)
                and t.amount > tv
                and (not exact_bridge_only or is_exact_bridge_transfer(t, bridge))
            ]
            brg_trap = [t for t in tr if t1 <= t.time <= t2 and bridge not in t.people and t.amount > tv]
            if trap_mode == "dual" and (not win_trap or not brg_trap):
                continue
            if trap_mode == "single" and (not win_trap and not brg_trap):
                continue
            used_events = {t.event_id for t in in_set} | {t.event_id for t in win_trap[:2]} | {t.event_id for t in brg_trap[:2]}
            anc = [k for k in single_ticket[bridge] if k.event_id not in used_events]
            if not anc:
                continue
            anchor = max(anc, key=lambda k: len(k.label))
            # Target row must be unique; answer is the amount for stable grading.
            if sum(1 for t in in_set if t.amount == tv) != 1:
                continue
            brg_sorted = sorted(brg_trap, key=lambda t: t.amount)
            win_sorted = sorted(win_trap, key=lambda t: t.amount)
            counter = [k for k in all_single_tickets if k.people[0] != bridge and k.event_id not in used_events]
            rng.shuffle(counter)
            priority = brg_sorted[:2] + win_sorted[:2]
            other_fill = [t for t in tr if t not in in_set and t not in win_trap and t not in brg_trap]
            rng.shuffle(other_fill)

            pos_docs, seen = [], set()
            for d in [anchor.doc] + [t.doc for t in in_set]:
                if d not in seen:
                    seen.add(d); pos_docs.append(d)
            neg_docs = []
            for d in [t.doc for t in priority] + [k.doc for k in counter[:2]] + [t.doc for t in other_fill]:
                if d in seen:
                    continue
                seen.add(d); neg_docs.append(d)
                if len(neg_docs) >= max_negatives:
                    break
            if any(gold in d for d in neg_docs) or gold in anchor.doc or anchor.label in target.doc:
                continue

            samples.append({
                "query": (
                    f"在我的 MobileMem 私人记忆里，有一张行程车票：{anchor.label}，"
                    f"那张车票的“同行好友”字段只有一位好友，把这位好友记为 TA。"
                    f"请先从车票文档读出 TA，再在 {win_label} 内筛选目标记录。"
                    f"目标记录必须同时满足：记录类型是“转账记录”；“同行好友”字段正好只有 TA；"
                    f"转账时间在上述时间段内。按这些条件共应筛出 2 条目标转账记录。"
                    f"不要把车票当成转账记录，也不要统计同行好友字段包含其他人的转账。"
                    f"请比较这 2 条目标转账记录的转账金额，只回答这 2 条目标转账记录中的最高转账金额；"
                    f"判断标准只有转账金额大小，不是转账时间早晚。"
                    f"注意：不要计入 TA 以外的人的转账，也不要计入不在该时间段内的转账，"
                    f"即使主题相近或金额更高。"
                ),
                "answer": [gold],
                "positive": pos_docs,
                "negative": neg_docs,
                "positive_wrong": [],
                "fakeanswer": "",
                "mobilemem_meta": {
                    "uid": uid, "question_mode": "multihop_bridge_money",
                    "hardness": "bridge_window_max", "bridge_person": bridge,
                    "anchor_kind": "ticket", "anchor_entity": anchor.label,
                    "target_kind": "money", "aggregate": "max",
                    "answer_mode": "max_amount",
                    "window_label": win_label,
                    "window": [f"{t1:%Y-%m-%d %H:%M}", f"{t2:%Y-%m-%d %H:%M}"],
                    "gold_entity": target.desc, "gold_value": tv,
                    "runner_up_value": runner_up,
                    "gold_time": f"{target.time:%Y-%m-%d %H:%M}",
                    "runner_up_time": f"{min((t for t in in_set if t is not target), key=lambda t: t.amount).time:%Y-%m-%d %H:%M}",
                    "gold_is_latest_target": target.time == max(t.time for t in in_set),
                    "positive_margin": round(tv - runner_up, 3),
                    "positive_ratio": round(tv / max(runner_up, 1e-9), 3),
                    "in_window_count": len(in_set),
                    "window_trap_value": min((t.amount for t in win_trap), default=None),
                    "bridge_trap_value": min((t.amount for t in brg_trap), default=None),
                    "priority_traps": len(priority),
                    "n_positive": len(pos_docs), "n_negative": len(neg_docs),
                    "exact_bridge_only": exact_bridge_only,
                    "generation_filter": (
                        "dual_trap_exact_two_exact_bridge_margin_no_behavior_filter"
                    ),
                },
            })
            used.add((bridge, gold))
            made += 1
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "20.zip")
    ap.add_argument("--uids", type=str, default="0-9")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-id", type=int, default=360000)
    ap.add_argument("--max-negatives", type=int, default=40)
    ap.add_argument("--per-bridge", type=int, default=3)
    ap.add_argument("--min-in-window", type=int, default=2)
    ap.add_argument("--trap-mode", choices=("dual", "single"), default="dual")
    ap.add_argument("--exact-in-window", type=int, default=2)
    ap.add_argument(
        "--exact-bridge-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require target/window-trap transfers to have exactly one companion: TA.",
    )
    ap.add_argument("--min-positive-margin", type=float, default=50.0)
    ap.add_argument("--min-positive-ratio", type=float, default=1.2)
    ap.add_argument("--min-gold-amount", type=float, default=250.0)
    ap.add_argument(
        "--prefer-gold-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort clean-stable candidates where the max-amount target is also the latest target first.",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--preview-limit", type=int, default=10, help="0 = do not write preview")
    ap.add_argument(
        "--preview-output",
        type=Path,
        default=ROOT / "data" / "rgb" / "zh_mobilemem_multihop_preview.json",
    )
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "rgb" / "zh_mobilemem_multihop.json")
    args = ap.parse_args()

    uids = []
    for part in args.uids.split(","):
        if "-" in part:
            a, b = part.split("-"); uids += list(range(int(a), int(b) + 1))
        elif part.strip():
            uids.append(int(part))

    data_dir, _tmp = _open_mobilemem_root(args.input)
    owners = load_owners(data_dir)
    transfers = load_transfers(data_dir, owners)
    tickets = load_tickets(data_dir, owners)
    wins = calendar_windows()
    samples: list[dict[str, Any]] = []
    for uid in uids:
        rng = random.Random(args.seed + uid)
        samples += make_samples(transfers, tickets, rng, uid=uid, windows=wins,
                                min_in=args.min_in_window, max_negatives=args.max_negatives,
                                per_bridge=args.per_bridge, trap_mode=args.trap_mode,
                                exact_bridge_only=args.exact_bridge_only,
                                exact_in_window=args.exact_in_window,
                                min_positive_margin=args.min_positive_margin,
                                min_positive_ratio=args.min_positive_ratio,
                                min_gold_amount=args.min_gold_amount,
                                prefer_gold_latest=args.prefer_gold_latest)
    samples.sort(
        key=lambda s: (
            0 if (
                args.prefer_gold_latest
                and s["mobilemem_meta"].get("gold_is_latest_target", False)
            ) else 1,
            -s["mobilemem_meta"].get("positive_margin", 0),
            -s["mobilemem_meta"].get("priority_traps", 0),
            -s["mobilemem_meta"].get("gold_value", 0),
        ),
    )
    if args.limit:
        samples = samples[: args.limit]
    for i, s in enumerate(samples):
        s["id"] = args.start_id + i

    out = args.output if args.output.is_absolute() else ROOT / args.output
    write_jsonl(out, samples)
    summary = audit_money(samples)
    summary["uids"] = uids
    out.with_suffix(".audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"output": str(out), **summary}
    if args.preview_limit:
        preview = samples[: args.preview_limit]
        preview_out = args.preview_output if args.preview_output.is_absolute() else ROOT / args.preview_output
        write_jsonl(preview_out, preview)
        preview_summary = audit_money(preview)
        preview_out.with_suffix(".audit.json").write_text(
            json.dumps(preview_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["preview_output"] = str(preview_out)
        result["preview_rows"] = len(preview)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
