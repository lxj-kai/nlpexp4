"""Download NoiserBench JSON files (incl. Git LFS) into data/noiser_bench/.

Source: https://huggingface.co/datasets/Jinyang23/NoiserBench
Also vendored on origin/main under data/noiser_bench/.

Usage:
    python scripts/prepare_noiser_bench.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

HF_REPO = "Jinyang23/NoiserBench"
OUT_DIR = ROOT / "data" / "noiser_bench"

_FILES = (
    "single-hop/nq.json",
    "single-hop/rgb.json",
    "multi-hop/explicit/hotpotqa.json",
    "multi-hop/explicit/2wikimqa.json",
    "multi-hop/explicit/bamboogle.json",
    "multi-hop/implicit/strategyqa.json",
    "multi-hop/implicit/tempqa.json",
    "mix-hop/priorqa.json",
)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    head = path.read_text(encoding="utf-8", errors="ignore")[:64]
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _validate_json(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return len(data)


def prepare(*, download: bool = True) -> dict[str, int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    if download:
        from huggingface_hub import hf_hub_download

        for rel in _FILES:
            target = OUT_DIR / rel
            if target.exists() and not _is_lfs_pointer(target):
                logger.info(f"skip existing {rel}")
                continue
            logger.info(f"downloading {HF_REPO}/{rel} ...")
            downloaded = hf_hub_download(
                HF_REPO,
                rel,
                repo_type="dataset",
                local_dir=str(OUT_DIR),
            )
            src = Path(downloaded)
            target.parent.mkdir(parents=True, exist_ok=True)
            if src != target:
                target.write_bytes(src.read_bytes())

    for rel in _FILES:
        path = OUT_DIR / rel
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run with download enabled")
        if _is_lfs_pointer(path):
            raise FileNotFoundError(
                f"{path} is still a Git LFS pointer; install git-lfs or rerun this script"
            )
        key = rel.split("/")[-1].replace(".json", "")
        counts[key] = _validate_json(path)
        logger.info(f"  {rel}: {counts[key]} records")

    return counts


def main() -> None:
    counts = prepare(download=True)
    print(json.dumps({"dir": str(OUT_DIR), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
