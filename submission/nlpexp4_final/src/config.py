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


def normalize_openai_base(url: str) -> str:
    """LM Studio 等服务地址统一补全 /v1 后缀。"""
    base = (url or "").strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:1234/v1"
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base

_DATASET_DIRS: dict[str, Path] = {
    "rgb": PROJECT_ROOT / "data" / "rgb",
    "miriad": PROJECT_ROOT / "data" / "miriad",
    "cmedqa": PROJECT_ROOT / "data" / "cmedqa",
    "2wiki": PROJECT_ROOT / "data" / "2wiki",
    "bright": PROJECT_ROOT / "data" / "bright",
    "multihop_rag": PROJECT_ROOT / "data" / "multihop_rag",
    "tempo": PROJECT_ROOT / "data" / "tempo",
    "noiser_bench": PROJECT_ROOT / "data" / "noiser_bench",
    "mobilemem": PROJECT_ROOT / "data" / "mobilemem",
}


def resolve_data_dir(dataset: str | None = None) -> Path:
    """Resolve dataset directory from explicit name or env override."""
    if dataset:
        key = dataset.strip().lower()
        if key not in _DATASET_DIRS:
            raise ValueError(f"unknown dataset: {dataset!r}; choose from {sorted(_DATASET_DIRS)}")
        return _DATASET_DIRS[key]
    env_dir = os.getenv("NLP4_DATA_DIR")
    if env_dir:
        return PROJECT_ROOT / env_dir if not Path(env_dir).is_absolute() else Path(env_dir)
    env_dataset = os.getenv("NLP4_DATASET", "rgb").strip().lower()
    return _DATASET_DIRS.get(env_dataset, _DATASET_DIRS["rgb"])


@dataclass(frozen=True)
class Config:
    """全局只读配置（dataclass + frozen 防误改）。"""

    # ── LM Studio（问答生成）──
    api_key: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    api_base: str = normalize_openai_base(
        os.getenv("LMSTUDIO_API_BASE", "http://127.0.0.1:1234")
    )
    model: str = os.getenv("LMSTUDIO_MODEL", "qwen2.5-0.5b-instruct-mlx")

    # ── DeepSeek（审查 Judge，与生成模型分离）──
    judge_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    judge_api_base: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    judge_model: str = os.getenv(
        "DEEPSEEK_JUDGE_MODEL",
        os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    context_length: int = int(os.getenv("NLP4_CONTEXT_LENGTH", "8192"))
    context_reserve_tokens: int = int(os.getenv("NLP4_CONTEXT_RESERVE", "64"))
    max_query_chars: int = int(os.getenv("NLP4_MAX_QUERY_CHARS", "4000"))
    max_context_chars: int = int(os.getenv("NLP4_MAX_CONTEXT_CHARS", "8000"))

    # ── 路径 ──
    project_root: Path = PROJECT_ROOT
    dataset: str = os.getenv("NLP4_DATASET", "rgb")
    data_dir: Path = field(default_factory=lambda: resolve_data_dir())
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
        d.pop("judge_api_key", None)
        return d


CONFIG: Config = Config()
