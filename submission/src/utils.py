"""Lightweight shared utilities (logging / IO / random)."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s - %(message)s", "%H:%M:%S"
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)


def parallel_map(
    items: list[T],
    fn: Callable[[T], R],
    *,
    workers: int = 1,
    show_progress: bool = True,
    desc: str = "",
) -> list[R]:
    """Parallel map preserving input order. workers=1 runs sequentially."""
    if not items:
        return []
    if workers <= 1:
        if show_progress:
            from tqdm import tqdm

            return [fn(x) for x in tqdm(items, desc=desc, disable=not show_progress)]
        return [fn(x) for x in items]

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        iterator = as_completed(future_to_idx)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(items), desc=desc, disable=not show_progress)
        for future in iterator:
            idx = future_to_idx[future]
            results[idx] = future.result()
    return [r for r in results if r is not None]


class Timer:
    """简单的上下文计时器。"""

    def __init__(self, name: str = "task", logger: logging.Logger | None = None) -> None:
        self.name = name
        self.logger = logger or get_logger("timer")
        self._t0: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self._t0
        self.logger.info(f"[{self.name}] elapsed {self.elapsed:.2f}s")
