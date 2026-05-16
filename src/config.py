"""Global configuration for nlpexp4.

集中所有可调参数；其它模块统一从此读取，避免散落在代码各处。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """全局只读配置（dataclass + frozen 防误改）。"""

    # ── API ──
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    judge_model: str = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 512

    # ── 路径 ──
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / os.getenv("NLP4_DATA_DIR", "data/rgb")
    results_dir: Path = PROJECT_ROOT / os.getenv("NLP4_RESULTS_DIR", "experiments/results")
    cache_dir: Path = PROJECT_ROOT / os.getenv("NLP4_CACHE_DIR", ".cache")
    figures_dir: Path = PROJECT_ROOT / "figures"
    report_dir: Path = PROJECT_ROOT / "report"

    # ── 数据集文件名 ──
    zh_main: str = "zh.json"
    zh_fact: str = "zh_fact.json"
    zh_int: str = "zh_int.json"
    zh_refine: str = "zh_refine.json"
    en_main: str = "en.json"
    en_fact: str = "en_fact.json"

    # ── 噪音实验 ──
    noise_ratios: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    noise_types: Tuple[str, ...] = ("semantic", "counterfactual", "mixed")
    noise_positions: Tuple[str, ...] = ("front", "back", "interleave", "surround")
    max_docs: int = 10
    min_positive_docs: int = 1

    # ── 实验/复现 ──
    seed: int = int(os.getenv("NLP4_SEED", "42"))
    smoke_test_size: int = 50
    judge_score_max: int = 5

    # ── 元数据 ──
    languages: Tuple[str, ...] = ("zh", "en")
    correctors_enabled: Tuple[str, ...] = ("prompt", "iterative", "confidence", "selfrag")

    # ── 工具方法 ──
    def ensure_dirs(self) -> None:
        """确保所有输出目录存在。"""
        for d in (self.results_dir, self.cache_dir, self.figures_dir, self.report_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """序列化为可写盘的 dict（去掉 api_key，Path 转 str）。"""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        d.pop("api_key", None)
        return d


CONFIG: Config = Config()
