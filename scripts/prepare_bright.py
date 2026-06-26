"""Download xlangai/BRIGHT and convert to RGB-compatible JSONL.

Source: https://huggingface.co/datasets/xlangai/BRIGHT

BRIGHT has 12 subdomain splits. Each example links ``gold_ids`` (positives) and
often ``excluded_ids`` (human-verified hard negatives). ``gold_answer`` is a long
reasoning-style reference answer.

Usage:
    python scripts/prepare_bright.py
    python scripts/prepare_bright.py --no-snapshot
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
from src.data_loader import is_usable_gold_answer  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = ROOT / "data" / "bright"
DEFAULT_RAW_DIR = DEFAULT_OUT_DIR / "raw"
GOLD_FILL_CACHE = DEFAULT_OUT_DIR / "gold_fill_cache.json"
HF_REPO = "xlangai/BRIGHT"


def _load_gold_fill_cache() -> dict[str, str]:
    if not GOLD_FILL_CACHE.exists():
        return {}
    return json.loads(GOLD_FILL_CACHE.read_text(encoding="utf-8"))


def _resolve_gold_answer(row: dict, cache: dict[str, str]) -> str | None:
    answer = str(row.get("gold_answer", "")).strip()
    if is_usable_gold_answer(answer):
        return answer
    subdomain = str(row.get("_subdomain", ""))
    example_id = row.get("id")
    key = f"{subdomain}:{example_id}"
    filled = cache.get(key, "").strip()
    return filled if is_usable_gold_answer(filled) else None


def _flip_answer(answer: str) -> str:
    low = answer.strip().lower()
    if low == "yes":
        return "no"
    if low == "no":
        return "yes"
    if len(answer) > 80:
        return answer[:77].rstrip() + "..."
    return f"not {answer}"


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(records)} records -> {path}")


def _snapshot_raw(raw_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    marker = raw_dir / ".snapshot_complete"
    if marker.exists():
        logger.info(f"HF snapshot already present under {raw_dir}")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"snapshot_download {HF_REPO} -> {raw_dir} (may take a while) ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    marker.write_text("ok\n", encoding="utf-8")


def _load_subdomain_splits() -> list[str]:
    from datasets import get_dataset_split_names

    return list(get_dataset_split_names(HF_REPO, "examples"))


def _load_doc_map(subdomain: str) -> dict[str, str]:
    from datasets import load_dataset

    logger.info(f"loading documents/{subdomain} ...")
    ds = load_dataset(HF_REPO, "documents", split=subdomain)
    doc_map: dict[str, str] = {}
    for row in ds:
        text = str(row.get("content", "")).strip()
        if text:
            doc_map[str(row["id"])] = text
    logger.info(f"  {subdomain}: {len(doc_map)} documents")
    return doc_map


def _resolve_docs(doc_ids: list[str], doc_map: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        text = doc_map.get(str(doc_id), "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _pick_negatives(
    *,
    excluded_ids: list[str],
    gold_ids: set[str],
    doc_map: dict[str, str],
    n_neg: int,
    rng: random.Random,
) -> list[str]:
    hard = _resolve_docs(
        [x for x in excluded_ids if x and x != "N/A"],
        doc_map,
    )
    if len(hard) >= n_neg:
        return hard[:n_neg]

    pool_ids = [k for k in doc_map if k not in gold_ids]
    rng.shuffle(pool_ids)
    filler: list[str] = []
    seen = set(hard)
    for doc_id in pool_ids:
        text = doc_map[doc_id].strip()
        if text and text not in seen:
            filler.append(text)
            seen.add(text)
        if len(hard) + len(filler) >= n_neg:
            break
    return (hard + filler)[:n_neg]


def _load_examples(subdomain: str) -> list[dict]:
    from datasets import load_dataset

    logger.info(f"loading examples/{subdomain} ...")
    ds = load_dataset(HF_REPO, "examples", split=subdomain)
    rows = [dict(row) for row in ds]
    for row in rows:
        row["_subdomain"] = subdomain
    logger.info(f"  {subdomain}: {len(rows)} examples")
    return rows


def build_main_records(
    rows: list[dict],
    doc_maps: dict[str, dict[str, str]],
    *,
    size: int | None,
    n_neg: int,
    seed: int,
    gold_fill: dict[str, str] | None = None,
) -> list[dict]:
    cache = gold_fill or {}
    rng = random.Random(seed)
    if size is not None and size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        subdomain = str(row["_subdomain"])
        doc_map = doc_maps[subdomain]
        gold_ids = {str(x) for x in row.get("gold_ids") or []}
        positives = _resolve_docs(list(gold_ids), doc_map)
        if not positives:
            continue
        negatives = _pick_negatives(
            excluded_ids=list(row.get("excluded_ids") or []),
            gold_ids=gold_ids,
            doc_map=doc_map,
            n_neg=n_neg,
            rng=rng,
        )
        if not negatives:
            continue
        answer = _resolve_gold_answer(row, cache)
        if not answer:
            continue
        meta = {
            "subdomain": subdomain,
            "example_id": row.get("id"),
            "source": HF_REPO,
        }
        if str(row.get("gold_answer", "")).strip().upper() in ("N/A", "NA"):
            meta["gold_synthetic"] = True
        out.append(
            {
                "id": rec_id,
                "query": str(row.get("query", "")).strip(),
                "answer": [answer],
                "positive": positives,
                "negative": negatives,
                "meta": meta,
            }
        )
    return out


def build_fact_records(
    rows: list[dict],
    doc_maps: dict[str, dict[str, str]],
    *,
    size: int | None,
    n_neg: int,
    seed: int,
    gold_fill: dict[str, str] | None = None,
) -> list[dict]:
    cache = gold_fill or {}
    rng = random.Random(seed + 1)
    if size is not None and size < len(rows):
        rows = rng.sample(rows, size)

    out: list[dict] = []
    for rec_id, row in enumerate(rows):
        subdomain = str(row["_subdomain"])
        doc_map = doc_maps[subdomain]
        gold_ids = {str(x) for x in row.get("gold_ids") or []}
        positives = _resolve_docs(list(gold_ids), doc_map)
        if not positives:
            continue
        negatives = _pick_negatives(
            excluded_ids=list(row.get("excluded_ids") or []),
            gold_ids=gold_ids,
            doc_map=doc_map,
            n_neg=n_neg + 1,
            rng=rng,
        )
        if len(negatives) < 2:
            continue
        wrong_doc = rng.choice(negatives)
        negatives = [d for d in negatives if d != wrong_doc]
        if not negatives:
            continue
        answer = _resolve_gold_answer(row, cache)
        if not answer:
            continue
        meta = {
            "subdomain": subdomain,
            "example_id": row.get("id"),
            "source": HF_REPO,
        }
        if str(row.get("gold_answer", "")).strip().upper() in ("N/A", "NA"):
            meta["gold_synthetic"] = True
        out.append(
            {
                "id": rec_id,
                "query": str(row.get("query", "")).strip(),
                "answer": answer,
                "fakeanswer": _flip_answer(answer),
                "positive": positives[:1],
                "positive_wrong": [wrong_doc],
                "negative": negatives[:n_neg],
                "meta": meta,
            }
        )
    return out


def prepare(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    main_size: int | None = None,
    fact_size: int | None = None,
    n_neg: int = 8,
    seed: int = CONFIG.seed,
    snapshot: bool = True,
) -> dict[str, Path]:
    if snapshot:
        _snapshot_raw(raw_dir)

    subdomains = _load_subdomain_splits()
    doc_maps = {sp: _load_doc_map(sp) for sp in subdomains}
    rows: list[dict] = []
    for sp in subdomains:
        rows.extend(_load_examples(sp))
    logger.info(f"total BRIGHT examples: {len(rows)}")

    gold_fill = _load_gold_fill_cache()
    if gold_fill:
        logger.info(f"loaded {len(gold_fill)} synthetic gold entries from cache")

    main_records = build_main_records(
        rows, doc_maps, size=main_size, n_neg=n_neg, seed=seed, gold_fill=gold_fill
    )
    fact_records = build_fact_records(
        rows, doc_maps, size=fact_size, n_neg=n_neg, seed=seed, gold_fill=gold_fill
    )

    main_path = out_dir / "en.json"
    fact_path = out_dir / "en_fact.json"
    _write_jsonl(main_records, main_path)
    _write_jsonl(fact_records, fact_path)

    meta = {
        "source": HF_REPO,
        "subdomains": subdomains,
        "raw_examples": len(rows),
        "main_size": len(main_records),
        "fact_size": len(fact_records),
        "n_neg": n_neg,
        "seed": seed,
        "neg_strategy": "excluded_ids hard negatives + same-subdomain distractors",
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"main": main_path, "fact": fact_path, "meta": meta_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare BRIGHT for nlpexp4")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--main-size", type=int, default=None)
    p.add_argument("--fact-size", type=int, default=None)
    p.add_argument("--n-neg", type=int, default=8)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="skip full HF snapshot (still loads via datasets cache)",
    )
    args = p.parse_args()
    paths = prepare(
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        main_size=args.main_size,
        fact_size=args.fact_size,
        n_neg=args.n_neg,
        seed=args.seed,
        snapshot=not args.no_snapshot,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
