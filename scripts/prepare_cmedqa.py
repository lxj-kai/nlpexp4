"""Download C-MTEB/CmedqaRetrieval and convert to RGB-compatible JSONL.

Dataset: https://huggingface.co/datasets/C-MTEB/CmedqaRetrieval
Qrels:   https://huggingface.co/datasets/C-MTEB/CmedqaRetrieval-qrels

Each query links to 1+ relevant corpus passages (doctor answers). We treat
passage text as gold answer variants and sample unrelated corpus rows as noise.

Usage:
    python scripts/prepare_cmedqa.py
    python scripts/prepare_cmedqa.py --main-size 3999 --fact-size 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = ROOT / "data" / "cmedqa"
CORPUS_REPO = "C-MTEB/CmedqaRetrieval"
QRELS_REPO = "C-MTEB/CmedqaRetrieval-qrels"


def _load_source() -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    from datasets import load_dataset

    logger.info(f"loading {CORPUS_REPO} ...")
    queries = load_dataset(CORPUS_REPO, split="queries")
    corpus = load_dataset(CORPUS_REPO, split="corpus")
    logger.info(f"loading {QRELS_REPO} ...")
    qrels = load_dataset(QRELS_REPO, split="dev")

    qmap = {row["id"]: row["text"] for row in queries}
    cmap = {row["id"]: row["text"] for row in corpus}

    rel: dict[str, list[str]] = defaultdict(list)
    for row in qrels:
        if int(row["score"]) >= 1:
            rel[row["qid"]].append(row["pid"])

    logger.info(
        f"loaded queries={len(qmap)} corpus={len(cmap)} qrels={len(qrels)} "
        f"queries_with_pos={len(rel)}"
    )
    return qmap, cmap, rel


def _pick_negatives(
    *,
    qid: str,
    pos_ids: set[str],
    corpus_ids: list[str],
    n_neg: int,
    rng: random.Random,
) -> list[str]:
    pool = [cid for cid in corpus_ids if cid not in pos_ids]
    if len(pool) <= n_neg:
        chosen = pool
    else:
        chosen = rng.sample(pool, n_neg)
    return chosen


def build_main_records(
    qmap: dict[str, str],
    cmap: dict[str, str],
    rel: dict[str, list[str]],
    *,
    size: int,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    corpus_ids = list(cmap.keys())
    qids = sorted(rel.keys())
    if size < len(qids):
        qids = rng.sample(qids, size)
    else:
        qids = qids[:size]

    out: list[dict] = []
    for rec_id, qid in enumerate(qids):
        pos_ids = rel[qid]
        pos_texts = [cmap[pid] for pid in pos_ids if pid in cmap]
        if not pos_texts:
            continue
        neg_ids = _pick_negatives(
            qid=qid,
            pos_ids=set(pos_ids),
            corpus_ids=corpus_ids,
            n_neg=n_neg,
            rng=rng,
        )
        out.append(
            {
                "id": rec_id,
                "query": qmap[qid],
                "answer": pos_texts[:3],
                "positive": pos_texts,
                "negative": [cmap[cid] for cid in neg_ids],
                "meta": {
                    "qid": qid,
                    "pos_ids": pos_ids,
                    "source": CORPUS_REPO,
                },
            }
        )
    return out


def build_fact_records(
    qmap: dict[str, str],
    cmap: dict[str, str],
    rel: dict[str, list[str]],
    *,
    size: int,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed + 1)
    corpus_ids = list(cmap.keys())
    qids = sorted(rel.keys())
    if size < len(qids):
        qids = rng.sample(qids, size)
    else:
        qids = qids[:size]

    out: list[dict] = []
    for rec_id, qid in enumerate(qids):
        pos_ids = rel[qid]
        pos_texts = [cmap[pid] for pid in pos_ids if pid in cmap]
        if not pos_texts:
            continue

        wrong_candidates = [q for q in qids if q != qid]
        wrong_qid = rng.choice(wrong_candidates)
        wrong_pos = [cmap[pid] for pid in rel[wrong_qid] if pid in cmap]
        if not wrong_pos:
            continue

        neg_ids = _pick_negatives(
            qid=qid,
            pos_ids=set(pos_ids) | set(rel[wrong_qid]),
            corpus_ids=corpus_ids,
            n_neg=n_neg,
            rng=rng,
        )
        out.append(
            {
                "id": rec_id,
                "query": qmap[qid],
                "answer": pos_texts[0],
                "fakeanswer": wrong_pos[0],
                "positive": pos_texts[:1],
                "positive_wrong": wrong_pos[:1],
                "negative": [cmap[cid] for cid in neg_ids],
                "meta": {
                    "qid": qid,
                    "wrong_qid": wrong_qid,
                    "source": CORPUS_REPO,
                },
            }
        )
    return out


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(records)} records -> {path}")


def prepare(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    main_size: int = 3999,
    fact_size: int = 500,
    n_neg: int = 8,
    seed: int = CONFIG.seed,
) -> dict[str, Path]:
    qmap, cmap, rel = _load_source()
    if not rel:
        raise RuntimeError("no qrels found")

    main_records = build_main_records(
        qmap, cmap, rel, size=main_size, n_neg=n_neg, seed=seed
    )
    fact_records = build_fact_records(
        qmap, cmap, rel, size=fact_size, n_neg=n_neg, seed=seed
    )

    main_path = out_dir / "zh.json"
    fact_path = out_dir / "zh_fact.json"
    _write_jsonl(main_records, main_path)
    _write_jsonl(fact_records, fact_path)

    meta = {
        "source": CORPUS_REPO,
        "qrels": QRELS_REPO,
        "main_size": len(main_records),
        "fact_size": len(fact_records),
        "n_neg": n_neg,
        "seed": seed,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"main": main_path, "fact": fact_path, "meta": meta_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CmedqaRetrieval for nlpexp4")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--main-size", type=int, default=3999)
    p.add_argument("--fact-size", type=int, default=500)
    p.add_argument("--n-neg", type=int, default=8)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    args = p.parse_args()
    paths = prepare(
        out_dir=args.out_dir,
        main_size=args.main_size,
        fact_size=args.fact_size,
        n_neg=args.n_neg,
        seed=args.seed,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
