"""OpenAI-compatible LLM client (LM Studio) with retry + disk cache + token bookkeeping."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI, APIError, APITimeoutError, BadRequestError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import CONFIG
from .prompt_budget import fit_messages_to_budget
from .utils import get_logger

logger = get_logger(__name__)

_GEMMA4_PAT = re.compile(r"gemma[-_ ]?4", re.I)
_FINAL_ANSWER_PATS = (
    re.compile(
        r"(?:Final Answer|最终答案|Final answer)[^:\n]*[:：]\**\s*(.+?)(?:\n|$)",
        re.I | re.S,
    ),
    re.compile(
        r"Formulate the Final Answer[^:\n]*[:：]\**\s*(.+?)(?:\.\s*$|\n)",
        re.I | re.S,
    ),
)
_META_LINE_PAT = re.compile(
    r"^(?:\d+\.|\*+|---|\[|thinking process|analyze|identify|synthesize|formulate|self-correction)",
    re.I,
)


def is_gemma4_model(model: str) -> bool:
    """识别 Gemma 4 系列（如 google/gemma-4-e4b）。"""
    return bool(_GEMMA4_PAT.search(model or ""))


def _message_field(message: Any, attr: str, *, extra_keys: tuple[str, ...] = ()) -> str:
    val = getattr(message, attr, None)
    if val:
        return str(val).strip()
    extra = getattr(message, "model_extra", None) or {}
    for key in extra_keys:
        if extra.get(key):
            return str(extra[key]).strip()
    return ""


def _strip_gemma_answer_markup(text: str) -> str:
    t = text.strip().strip('"').strip("'")
    return re.sub(r"^\*+|\*+$", "", t).strip()


def extract_gemma4_content(message: Any) -> str:
    """Gemma 4 在 LM Studio 等后端可能把 thinking 放在 reasoning 字段、content 为空。"""
    content = _message_field(message, "content")
    if content:
        return content

    reasoning = _message_field(
        message,
        "reasoning_content",
        extra_keys=("reasoning_content", "reasoning"),
    )
    if not reasoning:
        return ""

    if "<channel|>" in reasoning:
        tail = reasoning.split("<channel|>")[-1].strip()
        if tail:
            return _strip_gemma_answer_markup(tail)

    for pat in _FINAL_ANSWER_PATS:
        m = pat.search(reasoning)
        if m:
            ans = _strip_gemma_answer_markup(m.group(1))
            if ans:
                return ans

    for ln in reversed([x.strip() for x in reasoning.splitlines() if x.strip()]):
        if len(ln) < 2 or _META_LINE_PAT.match(ln):
            continue
        return _strip_gemma_answer_markup(ln)

    return ""


class LLMUsage:
    """累计 token 使用统计（便于跑实验时盯成本）。"""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.calls: int = 0
        self._lock = threading.Lock()

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
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
    """LM Studio / OpenAI 兼容 API 调用入口。

    - 启用磁盘缓存，避免相同 prompt 重复调用
    - 自动指数退避重试
    - 内置 token 使用统计
    """

    def __init__(
        self,
        *,
        use_cache: bool = True,
        api_key: str | None = None,
        api_base: str | None = None,
        cache_subdir: str = "llm",
        default_model: str | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else CONFIG.api_key
        self._api_base = api_base if api_base is not None else CONFIG.api_base
        self.default_model = default_model or CONFIG.model
        self.client = OpenAI(
            api_key=self._api_key or "lm-studio",
            base_url=self._api_base,
            timeout=CONFIG.timeout,
        )
        self.usage = LLMUsage()
        self.use_cache = use_cache
        self.cache_dir = CONFIG.cache_dir / cache_subdir
        self._cache_lock = threading.Lock()
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        with self._cache_lock:
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
            with self._cache_lock:
                self._cache_path(key).write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as e:
            logger.debug(f"cache save failed: {e}")

    def _max_prompt_tokens(self, max_tokens: int) -> int:
        return max(
            256,
            CONFIG.context_length - max_tokens - CONFIG.context_reserve_tokens,
        )

    def _fit_messages(
        self, messages: list[dict], *, max_tokens: int
    ) -> tuple[list[dict], bool]:
        budget = self._max_prompt_tokens(max_tokens)
        fitted, truncated = fit_messages_to_budget(messages, max_prompt_tokens=budget)
        if truncated:
            logger.warning(
                "prompt truncated to fit context (budget=%s tokens, model=%s)",
                budget,
                self.default_model,
            )
        return fitted, truncated

    @staticmethod
    def _is_context_length_error(exc: BadRequestError) -> bool:
        msg = str(exc).lower()
        return (
            "context length" in msg
            or "maximum context" in msg
            or "tokens to keep" in msg
            or "too many tokens" in msg
        )

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
        msg = resp.choices[0].message
        if is_gemma4_model(model):
            content = extract_gemma4_content(msg)
            if not content and (msg.content or _message_field(msg, "reasoning_content", extra_keys=("reasoning_content", "reasoning"))):
                logger.warning(
                    "Gemma 4 returned empty extractable answer (model=%s); "
                    "check max_tokens or reasoning length",
                    model,
                )
        else:
            content = msg.content or ""
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
        model = model or self.default_model
        temperature = CONFIG.temperature if temperature is None else temperature
        max_tokens = max_tokens or CONFIG.max_tokens

        messages, _ = self._fit_messages(messages, max_tokens=max_tokens)

        key = _cache_key(model, messages, temperature, max_tokens)
        cached = self._load_cache(key)
        if cached is not None:
            if is_gemma4_model(model) and not (cached.get("content") or "").strip():
                cached = None
            else:
                cached["cached"] = True
                return cached

        budget = self._max_prompt_tokens(max_tokens)
        fitted = messages
        payload: dict | None = None
        for attempt in range(5):
            try:
                payload = self._raw_chat(
                    fitted,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except BadRequestError as exc:
                if not self._is_context_length_error(exc) or attempt >= 4:
                    raise
                budget = max(256, budget // 2)
                logger.warning(
                    "context overflow from API; retry %s with budget=%s tokens",
                    attempt + 1,
                    budget,
                )
                fitted, _ = fit_messages_to_budget(
                    messages, max_prompt_tokens=budget
                )
                key = _cache_key(model, fitted, temperature, max_tokens)
        if payload is None:
            raise RuntimeError("LLM chat failed after context budget retries")
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


_GLOBAL_JUDGE: LLMClient | None = None


def get_judge_client(*, use_cache: bool = True) -> LLMClient:
    """DeepSeek judge client — 与本地生成模型分离。"""
    global _GLOBAL_JUDGE
    if _GLOBAL_JUDGE is None:
        if not CONFIG.judge_api_key:
            logger.warning("DEEPSEEK_API_KEY 未配置，Judge 调用会失败")
        _GLOBAL_JUDGE = LLMClient(
            use_cache=use_cache,
            api_key=CONFIG.judge_api_key,
            api_base=CONFIG.judge_api_base,
            cache_subdir="judge",
            default_model=CONFIG.judge_model,
        )
    return _GLOBAL_JUDGE
