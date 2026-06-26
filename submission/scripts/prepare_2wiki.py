"""Download xanhho/2WikiMultihopQA and convert to RGB-compatible JSONL.

Source: https://huggingface.co/datasets/xanhho/2WikiMultihopQA

Each row has ~10 Wikipedia articles in ``context`` with 2–4 ``supporting_facts``.
Non-supporting articles are used as hard semantic negatives (same-context distractors).

Usage:
    python scripts/prepare_2wiki.py
    python scripts/prepare_2wiki.py --split dev --main-size 5000 --fact-size 800
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = ROOT / "data" / "2wiki"
DEFAULT_RAW_DIR = DEFAULT_OUT_DIR / "raw"
HF_REPO = "xanhho/2WikiMultihopQA"


def _parse_json_field(raw: object) -> object:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _article_text(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and s.strip())


def _context_map(context: list) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for title, sentences in context:
        out[str(title)] = [str(s) for s in sentences]
    return out


def _supporting_set(supporting_facts: list) -> set[tuple[str, int]]:
    return {(str(title), int(idx)) for title, idx in supporting_facts}


def _positive_docs(
    ctx_map: dict[str, list[str]], supporting_facts: list
) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for title, idx in supporting_facts:
        sents = ctx_map.get(str(title), [])
        if not sents:
            continue
        idx = max(0, min(int(idx), len(sents) - 1))
        text = sents[idx].strip()
        if text and text not in seen:
            docs.append(text)
            seen.add(text)
    if not docs:
        for title, idx in supporting_facts:
            text = _article_text(ctx_map.get(str(title), []))
            if text and text not in seen:
                docs.append(text)
                seen.add(text)
    return docs


def _negative_docs(
    ctx_map: dict[str, list[str]],
    supporting_facts: list,
    *,
    n_neg: int,
    rng: random.Random,
) -> list[str]:
    sup = _supporting_set(supporting_facts)
    sup_titles = {title for title, _ in sup}
    pool: list[str] = []

    for title, sents in ctx_map.items():
        if title not in sup_titles:
            text = _article_text(sents)
            if text:
                pool.append(text)
            continue
        for idx, sent in enumerate(sents):
            if (title, idx) not in sup:
                text = sent.strip()
                if text:
                    pool.append(text)

    if not pool:
        return []
    if len(pool) <= n_neg:
        return pool
    return rng.sample(pool, n_neg)


def _flip_answer(answer: str) -> str:
    low = answer.strip().lower()
    if low == "yes":
        return "no"
    if low == "no":
        return "yes"
    return f"not {answer}"


def build_main_records(
    rows: list[dict],
    *,
    size: int,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    if size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        context = _parse_json_field(row["context"])
        supporting_facts = _parse_json_field(row["supporting_facts"])
        ctx_map = _context_map(context)
        positives = _positive_docs(ctx_map, supporting_facts)
        if not positives:
            continue
        negatives = _negative_docs(ctx_map, supporting_facts, n_neg=n_neg, rng=rng)
        if not negatives:
            continue
        answer = str(row["answer"]).strip()
        out.append(
            {
                "id": rec_id,
                "query": str(row["question"]).strip(),
                "answer": [answer],
                "positive": positives,
                "negative": negatives,
                "meta": {
                    "wiki_id": row.get("_id", ""),
                    "type": row.get("type", ""),
                    "split": row.get("_split", ""),
                    "source": HF_REPO,
                },
            }
        )
    return out


def build_fact_records(
    rows: list[dict],
    *,
    size: int,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed + 1)
    if size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        context = _parse_json_field(row["context"])
        supporting_facts = _parse_json_field(row["supporting_facts"])
        ctx_map = _context_map(context)
        positives = _positive_docs(ctx_map, supporting_facts)
        if not positives:
            continue
        negatives = _negative_docs(ctx_map, supporting_facts, n_neg=n_neg, rng=rng)
        if len(negatives) < 2:
            continue

        wrong_doc = rng.choice(negatives)
        negatives = [d for d in negatives if d != wrong_doc]
        if not negatives:
            continue

        answer = str(row["answer"]).strip()
        out.append(
            {
                "id": rec_id,
                "query": str(row["question"]).strip(),
                "answer": answer,
                "fakeanswer": _flip_answer(answer),
                "positive": positives[:1],
                "positive_wrong": [wrong_doc],
                "negative": negatives[:n_neg],
                "meta": {
                    "wiki_id": row.get("_id", ""),
                    "type": row.get("type", ""),
                    "split": row.get("_split", ""),
                    "source": HF_REPO,
                },
            }
        )
    return out


def _load_parquet_rows(path: Path, *, split: str) -> list[dict]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        row["_split"] = split
    logger.info(f"loaded {len(rows)} rows from {path.name}")
    return rows


def _download_parquet(raw_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.parquet", "dev.parquet", "test.parquet"):
        target = raw_dir / name
        if target.exists():
            continue
        logger.info(f"downloading {HF_REPO}/{name} ...")
        hf_hub_download(
            HF_REPO,
            name,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(records)} records -> {path}")


def prepare(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    split: str = "dev",
    main_size: int = 5000,
    fact_size: int = 800,
    n_neg: int = 8,
    seed: int = CONFIG.seed,
    download: bool = True,
) -> dict[str, Path]:
    if download:
        _download_parquet(raw_dir)

    parquet_path = raw_dir / f"{split}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"missing {parquet_path}; run with --download or place parquet under {raw_dir}"
        )

    rows = _load_parquet_rows(parquet_path, split=split)
    main_records = build_main_records(rows, size=main_size, n_neg=n_neg, seed=seed)
    fact_records = build_fact_records(rows, size=fact_size, n_neg=n_neg, seed=seed)

    main_path = out_dir / "en.json"
    fact_path = out_dir / "en_fact.json"
    _write_jsonl(main_records, main_path)
    _write_jsonl(fact_records, fact_path)

    meta = {
        "source": HF_REPO,
        "split": split,
        "main_size": len(main_records),
        "fact_size": len(fact_records),
        "n_neg": n_neg,
        "seed": seed,
        "neg_strategy": "same-context distractor articles/sentences",
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"main": main_path, "fact": fact_path, "meta": meta_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare 2WikiMultihopQA for nlpexp4")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--main-size", type=int, default=5000)
    p.add_argument("--fact-size", type=int, default=800)
    p.add_argument("--n-neg", type=int, default=8)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    paths = prepare(
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        split=args.split,
        main_size=args.main_size,
        fact_size=args.fact_size,
        n_neg=args.n_neg,
        seed=args.seed,
        download=not args.no_download,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
