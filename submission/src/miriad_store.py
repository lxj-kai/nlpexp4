"""Local MIRIAD-5.8M storage, verification, and RGBRecord materialization."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from .config import PROJECT_ROOT
from .data_loader import Language, RGBRecord, Subset
from .utils import get_logger

logger = get_logger(__name__)

MIRIAD_REPO = "miriad/miriad-5.8M"
MIRIAD_RAW_DIR = PROJECT_ROOT / "data" / "miriad" / "raw"
EXPECTED_SHARDS = 64
EXPECTED_ROWS = 5_821_948
MIN_TOTAL_BYTES = 7_000_000_000  # ~7GB on HF

SubsetName = Literal["main", "fact"]


def list_parquet_shards(raw_dir: Path) -> list[Path]:
    shards = sorted((raw_dir / "data").glob("train-*-of-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no MIRIAD parquet shards under {raw_dir / 'data'}")
    return shards


def _count_rows(shards: list[Path]) -> tuple[int, dict[str, int]]:
    per_shard: dict[str, int] = {}
    total = 0
    for path in shards:
        n = pq.ParquetFile(path).metadata.num_rows
        per_shard[path.name] = n
        total += n
    return total, per_shard


def verify_installation(raw_dir: Path = MIRIAD_RAW_DIR) -> dict:
    shards = list_parquet_shards(raw_dir)
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"MIRIAD incomplete: found {len(shards)} shards, expected {EXPECTED_SHARDS}. "
            f"Run `python scripts/prepare_miriad.py`."
        )

    total_rows, per_shard = _count_rows(shards)
    if total_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"MIRIAD row count mismatch: found {total_rows}, expected {EXPECTED_ROWS}."
        )

    total_bytes = sum(path.stat().st_size for path in shards)
    if total_bytes < MIN_TOTAL_BYTES:
        raise RuntimeError(
            f"MIRIAD total size too small: {total_bytes} bytes (< {MIN_TOTAL_BYTES}). "
            "Download is incomplete."
        )

    return {
        "shards": len(shards),
        "rows": total_rows,
        "bytes": total_bytes,
        "per_shard_rows": per_shard,
    }


def build_manifest(raw_dir: Path = MIRIAD_RAW_DIR) -> dict:
    report = verify_installation(raw_dir)
    shards = list_parquet_shards(raw_dir)
    return {
        "source": MIRIAD_REPO,
        "raw_dir": str(raw_dir),
        "shards": [path.name for path in shards],
        "rows": report["rows"],
        "bytes": report["bytes"],
        "per_shard_rows": report["per_shard_rows"],
    }


def _load_hf_dataset(raw_dir: Path):
    from datasets import load_dataset

    pattern = str(raw_dir / "data" / "train-*.parquet")
    ds = load_dataset("parquet", data_files=pattern, split="train")
    if len(ds) != EXPECTED_ROWS:
        raise RuntimeError(f"MIRIAD HF load row mismatch: {len(ds)} != {EXPECTED_ROWS}")
    return ds


def _validate_row(row: dict) -> dict:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    passage = str(row["passage_text"]).strip()
    if not question or not answer or not passage:
        raise ValueError(f"invalid MIRIAD row qa_id={row.get('qa_id')!r}")
    return row


def _sample_negatives_from_full(
    ds,
    row: dict,
    *,
    n_neg: int,
    rng: random.Random,
    max_attempts: int = 200_000,
) -> list[str]:
    specialty = str(row["specialty"]).strip()
    if not specialty:
        raise ValueError(f"MIRIAD row {row.get('qa_id')!r} missing specialty")
    paper_id = str(row["paper_id"])
    picked: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(picked) < n_neg:
        if attempts >= max_attempts:
            raise ValueError(
                f"insufficient MIRIAD negatives for qa_id={row.get('qa_id')!r}: "
                f"need {n_neg}, got {len(picked)} in specialty={specialty!r} "
                f"after {max_attempts} attempts"
            )
        candidate = _validate_row(ds[rng.randrange(len(ds))])
        if str(candidate["specialty"]).strip() != specialty:
            attempts += 1
            continue
        if str(candidate["paper_id"]) == paper_id:
            attempts += 1
            continue
        passage = candidate["passage_text"]
        if passage in seen:
            attempts += 1
            continue
        seen.add(passage)
        picked.append(passage)
        attempts += 1
    return picked


def _rows_to_main_records(
    ds,
    target_rows: list[dict],
    *,
    n_neg: int,
    seed: int,
) -> list[RGBRecord]:
    rng = random.Random(seed)
    out: list[RGBRecord] = []
    for rec_id, row in enumerate(target_rows):
        row = _validate_row(row)
        negatives = _sample_negatives_from_full(ds, row, n_neg=n_neg, rng=rng)
        out.append(
            RGBRecord(
                id=rec_id,
                query=row["question"],
                answer=[row["answer"]],
                positive=[row["passage_text"]],
                negative=negatives,
                language="en",
                subset="main",
                dataset="miriad",
            )
        )
    return out


def _specialty_index(rows: list[dict]) -> dict[str, list[int]]:
    by_specialty: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        specialty = str(row["specialty"]).strip()
        if not specialty:
            raise ValueError(f"MIRIAD row {row.get('qa_id')!r} missing specialty")
        by_specialty[specialty].append(i)
    return by_specialty


def _rows_to_fact_records(
    ds,
    target_rows: list[dict],
    pool_rows: list[dict],
    *,
    n_neg: int,
    seed: int,
) -> list[RGBRecord]:
    rng = random.Random(seed + 1)
    by_specialty = _specialty_index(pool_rows)

    out: list[RGBRecord] = []
    for rec_id, row in enumerate(target_rows):
        row = _validate_row(row)
        specialty = str(row["specialty"]).strip()
        wrong_candidates = [j for j in by_specialty[specialty] if j != rec_id]
        if not wrong_candidates:
            raise ValueError(
                f"no counterfactual candidate for qa_id={row.get('qa_id')!r} "
                f"in specialty={specialty!r}"
            )
        wrong = _validate_row(pool_rows[rng.choice(wrong_candidates)])
        negatives = _sample_negatives_from_full(ds, row, n_neg=n_neg, rng=rng)
        out.append(
            RGBRecord(
                id=rec_id,
                query=row["question"],
                answer=[row["answer"]],
                positive=[row["passage_text"]],
                negative=negatives,
                positive_wrong=[wrong["passage_text"]],
                fakeanswer=str(wrong["answer"]),
                language="en",
                subset="fact",
                dataset="miriad",
            )
        )
    return out


def load_miriad_records(
    *,
    subset: SubsetName,
    limit: int,
    shuffle: bool,
    seed: int,
    n_neg: int = 8,
    raw_dir: Path = MIRIAD_RAW_DIR,
) -> list[RGBRecord]:
    if subset not in ("main", "fact"):
        raise ValueError(f"MIRIAD does not provide subset {subset!r}")

    verify_installation(raw_dir)
    ds = _load_hf_dataset(raw_dir)

    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(ds):
        raise ValueError(f"limit={limit} exceeds MIRIAD size {len(ds)}")

    pool_size = min(len(ds), max(limit * 20, limit + 50))
    indices = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    selected = indices[:limit]
    target_rows = [_validate_row(ds[i]) for i in selected]
    fact_pool_indices = indices[:pool_size]
    fact_pool_rows = [_validate_row(ds[i]) for i in fact_pool_indices]

    if subset == "main":
        return _rows_to_main_records(ds, target_rows, n_neg=n_neg, seed=seed)
    return _rows_to_fact_records(
        ds, target_rows, fact_pool_rows, n_neg=n_neg, seed=seed
    )


def iter_miriad_records(
    *,
    subset: SubsetName,
    limit: int,
    shuffle: bool,
    seed: int,
    n_neg: int = 8,
    raw_dir: Path = MIRIAD_RAW_DIR,
):
    records = load_miriad_records(
        subset=subset,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        n_neg=n_neg,
        raw_dir=raw_dir,
    )
    yield from records
