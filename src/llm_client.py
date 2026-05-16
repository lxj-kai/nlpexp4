"""Deepseek API client with retry + disk cache + token bookkeeping."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import CONFIG
from .utils import get_logger

logger = get_logger(__name__)


class LLMUsage:
    """累计 token 使用统计（便于跑实验时盯成本）。"""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def _cache_key(
    model: str, messages: list[dict], temperature: float, max_tokens: int
) -> str:
    payload = json.dumps(
        {"m": model, "msg": messages, "t": temperature, "mt": max_tokens},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class LLMClient:
    """Deepseek 调用入口。

    - 启用磁盘缓存，避免相同 prompt 重复花钱
    - 自动指数退避重试
    - 内置 token 使用统计
    """

    def __init__(self, *, use_cache: bool = True) -> None:
        if not CONFIG.api_key:
            logger.warning("DEEPSEEK_API_KEY 未配置，调用会失败")
        self.client = OpenAI(
            api_key=CONFIG.api_key or "EMPTY",
            base_url=CONFIG.api_base,
            timeout=CONFIG.timeout,
        )
        self.usage = LLMUsage()
        self.use_cache = use_cache
        self.cache_dir = CONFIG.cache_dir / "llm"
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, key: str, payload: dict) -> None:
        if not self.use_cache:
            return
        try:
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"cache save failed: {e}")

    @retry(
        stop=stop_after_attempt(CONFIG.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        reraise=True,
    )
    def _raw_chat(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.time() - t0
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "content": content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "latency": latency,
            "model": model,
        }

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        model = model or CONFIG.model
        temperature = CONFIG.temperature if temperature is None else temperature
        max_tokens = max_tokens or CONFIG.max_tokens

        key = _cache_key(model, messages, temperature, max_tokens)
        cached = self._load_cache(key)
        if cached is not None:
            cached["cached"] = True
            return cached

        payload = self._raw_chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        self.usage.add(payload["prompt_tokens"], payload["completion_tokens"])
        payload["cached"] = False
        self._save_cache(key, payload)
        return payload

    def generate(self, system: str, user: str, **kwargs) -> str:
        out = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return out["content"].strip()


_GLOBAL_CLIENT: LLMClient | None = None


def get_client(*, use_cache: bool = True) -> LLMClient:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = LLMClient(use_cache=use_cache)
    return _GLOBAL_CLIENT
