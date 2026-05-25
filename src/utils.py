"""Lightweight shared utilities (logging / IO / random)."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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
