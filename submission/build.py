#!/usr/bin/env python3
"""Build course deliverable into submission/dist/staging/, then sync_to_branch.sh copies to submission/."""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


SUBMISSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUBMISSION_DIR.parent
DEFAULT_MANIFEST = SUBMISSION_DIR / "manifest.yaml"
TEMPLATE_README = SUBMISSION_DIR / "templates" / "README.md"
PACKAGE_GITIGNORE = SUBMISSION_DIR / "templates" / "package.gitignore"
STAGING_DIR = SUBMISSION_DIR / "dist" / "staging"


def _load_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    raise RuntimeError("PyYAML required: pip install pyyaml")


def _matches_exclude(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        if "/" not in pat and fnmatch.fnmatch(Path(rel_posix).name, pat):
            return True
    return False


def _expand_include(repo_root: Path, item: str) -> list[Path]:
    if any(ch in item for ch in "*?[]"):
        matches = sorted(repo_root.glob(item))
        if not matches:
            print(f"  [skip no match] {item}")
        return [p for p in matches if p.exists()]
    src = repo_root / item
    if not src.exists():
        print(f"  [skip missing] {item}")
        return []
    return [src]


def _iter_sources(repo_root: Path, include: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for item in include:
        for src in _expand_include(repo_root, item):
            key = src.resolve().as_posix()
            if key not in seen:
                seen.add(key)
                found.append(src)
    return found


def _copy_file(src: Path, dst: Path, dry_run: bool, stats: dict) -> None:
    if dry_run:
        stats["files"] += 1
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    stats["files"] += 1
    stats["bytes"] += src.stat().st_size


def _copy_includes(
    repo_root: Path,
    out_root: Path,
    include: list[str],
    exclude_globs: list[str],
    dry_run: bool,
) -> dict:
    stats = {"files": 0, "bytes": 0, "excluded": 0}
    for src in _iter_sources(repo_root, include):
        if src.is_file():
            rel = src.relative_to(repo_root).as_posix()
            if _matches_exclude(rel, exclude_globs):
                stats["excluded"] += 1
                continue
            _copy_file(src, out_root / rel, dry_run, stats)
            continue

        for path in sorted(src.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(repo_root).as_posix()
            if _matches_exclude(rel, exclude_globs):
                stats["excluded"] += 1
                continue
            _copy_file(path, out_root / rel, dry_run, stats)
    return stats


def build(*, manifest_path: Path = DEFAULT_MANIFEST, dry_run: bool = False) -> Path:
    cfg = _load_manifest(manifest_path)
    include = cfg.get("include", [])
    exclude_globs = cfg.get("exclude_globs", [])

    print(f"Repository root : {REPO_ROOT}")
    print(f"Staging directory: {STAGING_DIR}")
    print(f"Dry run         : {dry_run}")
    print()

    if not dry_run:
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

    stats = _copy_includes(REPO_ROOT, STAGING_DIR, include, exclude_globs, dry_run)

    if dry_run:
        print(f"Would copy {stats['files']} files (excluded {stats['excluded']} paths)")
        return STAGING_DIR

    if TEMPLATE_README.exists():
        shutil.copy2(TEMPLATE_README, STAGING_DIR / "README.md")
    if PACKAGE_GITIGNORE.exists():
        shutil.copy2(PACKAGE_GITIGNORE, STAGING_DIR / ".gitignore")

    mb = stats["bytes"] / (1024 * 1024)
    print(f"Copied {stats['files']} files ({mb:.1f} MiB), excluded {stats['excluded']} paths")
    print(f"Next: bash submission/sync_to_branch.sh")
    return STAGING_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build submission/ deliverable staging tree")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        build(manifest_path=args.manifest.resolve(), dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
