"""实验结果路径工具：支持归档后的子目录递归查找。"""
from __future__ import annotations

from pathlib import Path

from .config import CONFIG, PROJECT_ROOT


def results_dir(custom: Path | str | None = None) -> Path:
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return CONFIG.results_dir


def glob_results(pattern: str, *, base: Path | None = None) -> list[Path]:
    """在 results 根目录及子目录中匹配 JSON（pattern 如 exp2_correction_zh_*.json）。"""
    root = base or results_dir()
    if "*" in pattern:
        return sorted(root.rglob(pattern))
    return sorted(root.rglob(f"{pattern}.json" if not pattern.endswith(".json") else pattern))
