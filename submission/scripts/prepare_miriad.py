"""Download the full MIRIAD-5.8M dataset into the project directory.

Source: https://huggingface.co/datasets/miriad/miriad-5.8M

This script MUST download all 64 parquet shards (~7.5GB) and verify row counts.
It does not create truncated JSONL subsets.

Usage:
    python scripts/prepare_miriad.py
    python scripts/prepare_miriad.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.miriad_store import (  # noqa: E402
    EXPECTED_ROWS,
    EXPECTED_SHARDS,
    MIRIAD_RAW_DIR,
    MIRIAD_REPO,
    build_manifest,
    list_parquet_shards,
    verify_installation,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_OUT_DIR = ROOT / "data" / "miriad"


def download_full_dataset(*, out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    from huggingface_hub import snapshot_download

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading full {MIRIAD_REPO} into {raw_dir} ...")
    snapshot_download(
        repo_id=MIRIAD_REPO,
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=["data/train-*.parquet", "README.md"],
    )
    return raw_dir


def prepare(*, out_dir: Path = DEFAULT_OUT_DIR, verify_only: bool = False) -> dict:
    raw_dir = out_dir / "raw"
    if not verify_only:
        download_full_dataset(out_dir=out_dir)

    report = verify_installation(raw_dir)
    manifest = build_manifest(raw_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"manifest -> {manifest_path}")
    return {"raw_dir": raw_dir, "manifest": manifest_path, "report": report}


def main() -> None:
    p = argparse.ArgumentParser(description="Download full MIRIAD-5.8M into project")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="only verify existing local parquet shards",
    )
    args = p.parse_args()
    result = prepare(out_dir=args.out_dir, verify_only=args.verify_only)
    print(
        json.dumps(
            {
                "raw_dir": str(result["raw_dir"]),
                "manifest": str(result["manifest"]),
                "report": result["report"],
                "expected_shards": EXPECTED_SHARDS,
                "expected_rows": EXPECTED_ROWS,
                "shards_found": len(list_parquet_shards(result["raw_dir"])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
