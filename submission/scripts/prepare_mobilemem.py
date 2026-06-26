"""Install MobileMem shopping-graph RAG JSONL into data/mobilemem/.

Source files originally live on origin/main under data/rgb/ with long names.
This script copies/renames them to the standalone MobileMem layout:

    data/mobilemem/zh_calc.json      (120 samples, multi-event aggregation)
    data/mobilemem/zh_noncalc.json   (120 samples, retrieval QA)

Usage:
    python scripts/prepare_mobilemem.py
    python scripts/prepare_mobilemem.py --from-git   # checkout from origin/main if missing
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "data" / "mobilemem"
LEGACY_RGB = ROOT / "data" / "rgb"
GIT_SOURCES = {
    "zh_calc.json": "data/rgb/zh_mobilemem_shopping_graph_hard_120.json",
    "zh_noncalc.json": "data/rgb/zh_mobilemem_shopping_graph_noncalc_hard_120.json",
}
LEGACY_LOCAL = {
    "zh_calc.json": LEGACY_RGB / "zh_mobilemem_shopping_graph_hard_120.json",
    "zh_noncalc.json": LEGACY_RGB / "zh_mobilemem_shopping_graph_noncalc_hard_120.json",
}


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _git_show(rev_path: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"origin/main:{rev_path}"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _materialize(name: str, *, from_git: bool) -> Path:
    dest = OUT_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"skip existing {dest}")
        return dest

    legacy = LEGACY_LOCAL[name]
    if legacy.exists():
        logger.info(f"copy legacy {legacy.name} -> {dest.name}")
        shutil.copy2(legacy, dest)
        return dest

    if from_git:
        rev = GIT_SOURCES[name]
        blob = _git_show(rev)
        if blob:
            dest.write_bytes(blob)
            logger.info(f"git checkout origin/main:{rev} -> {dest}")
            return dest

    raise FileNotFoundError(
        f"cannot find MobileMem source for {name}; "
        f"place file at {dest} or run with --from-git after git fetch origin"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--from-git",
        action="store_true",
        help="fetch missing files from origin/main via git show",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name in ("zh_calc.json", "zh_noncalc.json"):
        path = _materialize(name, from_git=args.from_git)
        sizes[name] = _count_lines(path)

    meta = {
        "source": "MobileMem shopping screenshot OCR memory graph (project internal)",
        "layout": {
            "calc": "zh_calc.json",
            "noncalc": "zh_noncalc.json",
        },
        "calc_size": sizes["zh_calc.json"],
        "noncalc_size": sizes["zh_noncalc.json"],
        "language": "zh",
        "evidence_type": "shopping_screenshot_ocr_text",
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"metadata -> {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
