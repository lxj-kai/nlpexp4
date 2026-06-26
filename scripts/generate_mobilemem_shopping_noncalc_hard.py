#!/usr/bin/env python3
"""Generate a hard non-arithmetic MobileMem shopping graph dataset."""

from __future__ import annotations

import itertools
import json
import random
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/rgb/zh_mobilemem_shopping_graph_hard_120.json"
OUT = ROOT / "data/rgb/zh_mobilemem_shopping_graph_noncalc_hard_120.json"

FIELD_RE = {
    "event_id": r"事件ID: (.+)",
    "event_name": r"事件名称: (.+)",
    "event_time": r"发生时间: (.+)",
    "parent_event_name": r"长期事件: (.+)",
    "product": r"商品名称: (.+)",
    "shop": r"店铺: (.+)",
    "price": r"实际价格: (.+)",
    "order_time": r"下单时间: (.+)",
    "rating": r"评分: (.+)",
}


def extract(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"missing field {pattern!r} in {text[:120]!r}")
    return m.group(1).strip()


def parse_doc(doc: str) -> dict:
    item = {k: extract(v, doc) for k, v in FIELD_RE.items()}
    item["doc"] = doc
    item["event_dt"] = datetime.strptime(item["event_time"], "%Y-%m-%d %H:%M")
    item["order_dt"] = datetime.strptime(item["order_time"], "%Y-%m-%d %H:%M:%S")
    item["month"] = item["event_dt"].month
    item["quarter"] = (item["month"] - 1) // 3 + 1
    return item


def fmt_scope(items: list[dict]) -> tuple[str, str]:
    months = {x["month"] for x in items}
    quarters = {x["quarter"] for x in items}
    if len(months) == 1:
        return "month", f"2025年{next(iter(months)):02d}月"
    if len(quarters) == 1:
        return "quarter", f"2025年第{next(iter(quarters))}季度"
    return "year", "2025年"


def enum_events(items: list[dict]) -> str:
    return "、".join(f"{x['event_time']}《{x['event_name']}》" for x in items)


def make_negative_meta(targets: list[dict], negatives: list[dict]) -> list[dict]:
    target_ids = {x["event_id"] for x in targets}
    target_parents = {x["parent_event_name"] for x in targets}
    target_months = {x["month"] for x in targets}
    return [
        {
            "source_event_id": n["event_id"],
            "source_event_name": n["event_name"],
            "source_artifact_type": "shopping",
            "same_parent_as_target": n["parent_event_name"] in target_parents,
            "same_month_as_target": n["month"] in target_months,
            "score": 50.0 if n["event_id"] not in target_ids and n["month"] in target_months else 35.0,
        }
        for n in negatives
    ]


def build_row(idx: int, items: list[dict], op: str, all_items: list[dict], rng: random.Random) -> dict:
    items = sorted(items, key=lambda x: (x["event_dt"], x["event_id"]))
    scope_type, scope_desc = fmt_scope(items)
    prefix = f"在{scope_desc}的私人记忆中，只统计以下{len(items)}个事件：{enum_events(items)}。根据这些事件各自的购物截图，"

    if op == "earliest_order_product":
        target = min(items, key=lambda x: (x["order_dt"], x["event_id"]))
        query = prefix + "哪条记录的下单时间最早？请只回答该记录的商品名称。"
        answer = target["product"]
        metric_name = "下单时间最早的商品名称"
    elif op == "latest_order_shop":
        target = max(items, key=lambda x: (x["order_dt"], x["event_id"]))
        query = prefix + "哪条记录的下单时间最晚？请只回答该记录的店铺名称。"
        answer = target["shop"]
        metric_name = "下单时间最晚的店铺名称"
    elif op == "latest_event_product":
        target = max(items, key=lambda x: (x["event_dt"], x["event_id"]))
        query = prefix + "发生时间最晚的事件对应的商品名称是什么？"
        answer = target["product"]
        metric_name = "发生时间最晚事件的商品名称"
    elif op == "event_to_shop":
        target = rng.choice(items)
        query = prefix + f"其中《{target['event_name']}》这条事件对应的店铺名称是什么？"
        answer = target["shop"]
        metric_name = "指定事件的店铺名称"
    elif op == "shop_to_product":
        target = rng.choice(items)
        query = prefix + f"其中店铺为“{target['shop']}”的记录，对应的商品名称是什么？"
        answer = target["product"]
        metric_name = "指定店铺的商品名称"
    elif op == "rating_event_name":
        target = rng.choice(items)
        query = prefix + f"评分为{target['rating']}且店铺为“{target['shop']}”的记录，来自哪个事件？请只回答事件名称。"
        answer = target["event_name"]
        metric_name = "指定评分和店铺的事件名称"
    else:
        raise ValueError(op)

    target_ids = {x["event_id"] for x in items}
    same_month = [x for x in all_items if x["event_id"] not in target_ids and x["month"] in {i["month"] for i in items}]
    same_parent = [x for x in all_items if x["event_id"] not in target_ids and x["parent_event_name"] in {i["parent_event_name"] for i in items}]
    rest = [x for x in all_items if x["event_id"] not in target_ids and x not in same_month and x not in same_parent]
    negatives = (same_parent + same_month + rest)[:]
    rng.shuffle(negatives)
    negatives = sorted(negatives, key=lambda x: (x["month"] not in {i["month"] for i in items}, x["event_id"]))[: min(26, len(negatives))]

    return {
        "id": 371000 + idx,
        "query": query,
        "answer": [answer],
        "positive": [x["doc"] for x in items],
        "negative": [x["doc"] for x in negatives],
        "positive_wrong": [],
        "fakeanswer": "",
        "mobilemem_meta": {
            "uid": idx,
            "event_ids": [x["event_id"] for x in items],
            "event_names": [x["event_name"] for x in items],
            "event_times": [x["event_dt"].strftime("%Y-%m-%d %H:%M:%S") for x in items],
            "parent_event_ids": [x["event_id"].split("_")[0] for x in items],
            "parent_event_names": [x["parent_event_name"] for x in items],
            "noise_policy": "real_cross_event_non_contradictory_non_supporting",
            "hardness": "graph_multi_event_multi_document_non_arithmetic",
            "question_mode": "enumerated",
            "target_condition": None,
            "scope_type": scope_type,
            "scope_desc": scope_desc,
            "operation": op,
            "metric": "shopping_non_numeric_attribute",
            "metric_name": metric_name,
            "answer_artifact_type": "shopping",
            "answer_source_key": None,
            "chat_sender": None,
            "required_positive_docs": len(items),
            "single_doc_answerable": False,
            "answer_is_derived": False,
            "requires_cross_event_comparison": op in {"earliest_order_product", "latest_order_shop", "latest_event_product"},
            "support_path": [
                {
                    "event_id": x["event_id"],
                    "event_name": x["event_name"],
                    "event_time": x["event_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                    "artifact_type": "shopping",
                    "product": x["product"],
                    "shop": x["shop"],
                    "order_time": x["order_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                    "rating": x["rating"],
                    "parent_event_id": x["event_id"].split("_")[0],
                    "parent_event_name": x["parent_event_name"],
                }
                for x in items
            ],
            "target_negative_docs": len(negatives),
            "actual_negative_docs": len(negatives),
            "note": "The question is scoped to enumerated target events. It requires locating and comparing non-numeric shopping attributes; no arithmetic is required.",
        },
        "negative_meta": make_negative_meta(items, negatives),
    }


def main() -> None:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    docs: dict[str, dict] = {}
    for row in rows:
        for doc in row["positive"] + row["negative"]:
            if "记录类型: 购物截图" not in doc:
                continue
            item = parse_doc(doc)
            docs[item["event_id"]] = item
    all_items = sorted(docs.values(), key=lambda x: (x["event_dt"], x["event_id"]))

    combos = []
    for quarter in range(1, 5):
        bucket = [x for x in all_items if x["quarter"] == quarter]
        combos.extend(itertools.combinations(bucket, 4))
    for month in sorted({x["month"] for x in all_items}):
        bucket = [x for x in all_items if x["month"] == month]
        if len(bucket) >= 3:
            combos.extend(itertools.combinations(bucket, min(4, len(bucket))))
    combos = list(dict.fromkeys(tuple(x["event_id"] for x in c) for c in combos))
    by_id = {x["event_id"]: x for x in all_items}

    rng = random.Random(20260626)
    rng.shuffle(combos)
    ops = [
        "earliest_order_product",
        "latest_order_shop",
        "latest_event_product",
        "event_to_shop",
        "shop_to_product",
        "rating_event_name",
    ]
    out = []
    for i in range(120):
        ids = combos[i % len(combos)]
        items = [by_id[eid] for eid in ids]
        out.append(build_row(i, items, ops[i % len(ops)], all_items, rng))

    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} rows to {OUT}")
    print(f"shopping source docs: {len(all_items)}")


if __name__ == "__main__":
    main()
