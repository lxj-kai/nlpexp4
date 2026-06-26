"""Download yixuantt/MultiHopRAG and convert to RGB-compatible JSONL.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG

Each row has explicit ``evidence_list`` (positive facts) and a news corpus for
hard negatives. Answers are short entity names suitable for multi-hop QA.

Usage:
    python scripts/prepare_multihop_rag.py
    python scripts/prepare_multihop_rag.py --main-size 2000 --fact-size 400
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = ROOT / "data" / "multihop_rag"
DEFAULT_RAW_DIR = DEFAULT_OUT_DIR / "raw"
HF_REPO = "yixuantt/MultiHopRAG"
QA_FILE = "MultiHopRAG.json"
CORPUS_FILE = "corpus.json"


def _flip_answer(answer: str) -> str:
    low = answer.strip().lower()
    if low == "yes":
        return "no"
    if low == "no":
        return "yes"
    return f"not {answer}"


def _download_raw(raw_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    qa_path = raw_dir / QA_FILE
    corpus_path = raw_dir / CORPUS_FILE
    for name, target in ((QA_FILE, qa_path), (CORPUS_FILE, corpus_path)):
        if target.exists():
            continue
        logger.info(f"downloading {HF_REPO}/{name} ...")
        downloaded = hf_hub_download(
            HF_REPO,
            name,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        src = Path(downloaded)
        if src != target and src.exists():
            src.replace(target)
    return qa_path, corpus_path


def _load_source(raw_dir: Path, *, download: bool) -> tuple[list[dict], list[dict]]:
    if download:
        qa_path, corpus_path = _download_raw(raw_dir)
    else:
        qa_path = raw_dir / QA_FILE
        corpus_path = raw_dir / CORPUS_FILE
    if not qa_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"missing raw files under {raw_dir}; run without --no-download"
        )
    with open(qa_path, encoding="utf-8") as f:
        rows = json.load(f)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"loaded qa={len(rows)} corpus={len(corpus)}")
    return rows, corpus


def _positive_texts(row: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ev in row.get("evidence_list") or []:
        fact = str(ev.get("fact", "")).strip()
        if not fact:
            title = str(ev.get("title", "")).strip()
            body = str(ev.get("body", "")).strip()
            fact = body or title
        if fact and fact not in seen:
            out.append(fact)
            seen.add(fact)
    return out


def _evidence_urls(row: dict) -> set[str]:
    urls: set[str] = set()
    for ev in row.get("evidence_list") or []:
        url = str(ev.get("url", "")).strip()
        if url:
            urls.add(url)
    return urls


def _corpus_bodies(corpus: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for doc in corpus:
        body = str(doc.get("body", "")).strip()
        if not body:
            continue
        url = str(doc.get("url", "")).strip()
        out.append((url, body))
    return out


def _pick_negatives(
    *,
    evidence_urls: set[str],
    corpus_pairs: list[tuple[str, str]],
    n_neg: int,
    rng: random.Random,
) -> list[str]:
    pool = [body for url, body in corpus_pairs if url not in evidence_urls]
    if not pool:
        pool = [body for _, body in corpus_pairs]
    if len(pool) <= n_neg:
        return pool
    return rng.sample(pool, n_neg)


def build_main_records(
    rows: list[dict],
    corpus: list[dict],
    *,
    size: int | None,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    corpus_pairs = _corpus_bodies(corpus)
    if size is not None and size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        positives = _positive_texts(row)
        if not positives:
            continue
        negatives = _pick_negatives(
            evidence_urls=_evidence_urls(row),
            corpus_pairs=corpus_pairs,
            n_neg=n_neg,
            rng=rng,
        )
        if not negatives:
            continue
        answer = str(row.get("answer", "")).strip()
        if not answer:
            continue
        out.append(
            {
                "id": rec_id,
                "query": str(row.get("query", "")).strip(),
                "answer": [answer],
                "positive": positives,
                "negative": negatives,
                "meta": {
                    "question_type": row.get("question_type", ""),
                    "source": HF_REPO,
                },
            }
        )
    return out


def build_fact_records(
    rows: list[dict],
    corpus: list[dict],
    *,
    size: int | None,
    n_neg: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed + 1)
    corpus_pairs = _corpus_bodies(corpus)
    if size is not None and size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        positives = _positive_texts(row)
        if not positives:
            continue
        negatives = _pick_negatives(
            evidence_urls=_evidence_urls(row),
            corpus_pairs=corpus_pairs,
            n_neg=n_neg + 1,
            rng=rng,
        )
        if len(negatives) < 2:
            continue
        wrong_doc = rng.choice(negatives)
        negatives = [d for d in negatives if d != wrong_doc]
        if not negatives:
            continue
        answer = str(row.get("answer", "")).strip()
        if not answer:
            continue
        out.append(
            {
                "id": rec_id,
                "query": str(row.get("query", "")).strip(),
                "answer": answer,
                "fakeanswer": _flip_answer(answer),
                "positive": positives[:1],
                "positive_wrong": [wrong_doc],
                "negative": negatives[:n_neg],
                "meta": {
                    "question_type": row.get("question_type", ""),
                    "source": HF_REPO,
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
    raw_dir: Path = DEFAULT_RAW_DIR,
    main_size: int | None = None,
    fact_size: int | None = None,
    n_neg: int = 8,
    seed: int = CONFIG.seed,
    download: bool = True,
) -> dict[str, Path]:
    rows, corpus = _load_source(raw_dir, download=download)
    main_records = build_main_records(
        rows, corpus, size=main_size, n_neg=n_neg, seed=seed
    )
    fact_records = build_fact_records(
        rows, corpus, size=fact_size, n_neg=n_neg, seed=seed
    )

    main_path = out_dir / "en.json"
    fact_path = out_dir / "en_fact.json"
    _write_jsonl(main_records, main_path)
    _write_jsonl(fact_records, fact_path)

    meta = {
        "source": HF_REPO,
        "main_size": len(main_records),
        "fact_size": len(fact_records),
        "n_neg": n_neg,
        "seed": seed,
        "neg_strategy": "corpus articles excluding evidence URLs",
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"main": main_path, "fact": fact_path, "meta": meta_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare MultiHopRAG for nlpexp4")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument(
        "--main-size",
        type=int,
        default=None,
        help="cap main subset (default: all QA rows)",
    )
    p.add_argument(
        "--fact-size",
        type=int,
        default=None,
        help="cap fact subset (default: all QA rows)",
    )
    p.add_argument("--n-neg", type=int, default=8)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    paths = prepare(
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        main_size=args.main_size,
        fact_size=args.fact_size,
        n_neg=args.n_neg,
        seed=args.seed,
        download=not args.no_download,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
