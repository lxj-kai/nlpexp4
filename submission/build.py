#!/usr/bin/env python3
"""Build a standalone submission package for course delivery.

Usage (from repository root or submission/):

    python submission/build.py
    python submission/build.py --zip
    python submission/build.py --dry-run

Output: submission/dist/nlpexp4_final/  (+ optional .zip)
The output tree is fully self-contained and does not reference the dev repo.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
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


def _load_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    data: dict = {"package_name": "nlpexp4_final", "include": [], "exclude_globs": []}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("- "):
            key = line[:-1]
            if key in ("include", "exclude_globs"):
                section = key
                data.setdefault(section, [])
            elif key == "package_name":
                section = "package_name"
            continue
        if line.startswith("- ") and section in ("include", "exclude_globs"):
            data[section].append(line[2:].strip())
        elif section == "package_name" and not line.startswith("-"):
            data["package_name"] = line
    return data


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(out_root: Path, stats: dict, built_at: str) -> None:
    files: list[dict] = []
    for path in sorted(out_root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(out_root).as_posix()
        files.append(
            {
                "path": rel,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    meta = {
        "package": out_root.name,
        "built_at": built_at,
        "file_count": len(files),
        "total_bytes": stats["bytes"],
        "files": files,
    }
    (out_root / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _render_readme(out_root: Path, built_at: str) -> None:
    if TEMPLATE_README.exists():
        text = TEMPLATE_README.read_text(encoding="utf-8")
        text = text.replace("{{BUILD_DATE}}", built_at[:10])
        text = text.replace("{{PACKAGE_NAME}}", out_root.name)
    else:
        text = f"# {out_root.name}\n\nBuilt at {built_at}\n"
    (out_root / "README.md").write_text(text, encoding="utf-8")


def _write_package_gitignore(out_root: Path) -> None:
    if PACKAGE_GITIGNORE.exists():
        shutil.copy2(PACKAGE_GITIGNORE, out_root / ".gitignore")


def _make_zip(out_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                arc = path.relative_to(out_root.parent).as_posix()
                zf.write(path, arc)


def build(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path | None = None,
    make_zip: bool = False,
    dry_run: bool = False,
) -> Path:
    cfg = _load_manifest(manifest_path)
    package_name = cfg.get("package_name", "nlpexp4_final")
    include = cfg.get("include", [])
    exclude_globs = cfg.get("exclude_globs", [])

    built_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    dist_dir = output_dir or (SUBMISSION_DIR / "dist")
    out_root = dist_dir / package_name

    print(f"Repository root : {REPO_ROOT}")
    print(f"Output directory: {out_root}")
    print(f"Dry run         : {dry_run}")
    print()

    if not dry_run:
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

    stats = _copy_includes(REPO_ROOT, out_root, include, exclude_globs, dry_run)

    if dry_run:
        print(f"Would copy {stats['files']} files (excluded {stats['excluded']} paths)")
        return out_root

    _render_readme(out_root, built_at)
    _write_package_gitignore(out_root)
    _write_manifest(out_root, stats, built_at)

    mb = stats["bytes"] / (1024 * 1024)
    print(f"Copied {stats['files']} files ({mb:.1f} MiB), excluded {stats['excluded']} paths")
    print(f"README -> {out_root / 'README.md'}")
    print(f"Manifest -> {out_root / 'SUBMISSION_MANIFEST.json'}")

    if make_zip:
        zip_path = dist_dir / f"{package_name}.zip"
        _make_zip(out_root, zip_path)
        zip_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Zip archive -> {zip_path} ({zip_mb:.1f} MiB)")

    return out_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build standalone nlpexp4 submission package")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to manifest.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override dist/ parent directory")
    parser.add_argument("--zip", action="store_true", help="Also create a .zip next to the output folder")
    parser.add_argument("--dry-run", action="store_true", help="Count files without copying")
    args = parser.parse_args(argv)

    try:
        build(
            manifest_path=args.manifest.resolve(),
            output_dir=args.output_dir,
            make_zip=args.zip,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
