"""Token budget helpers — fit prompts to the model context window."""
from __future__ import annotations

import functools
from typing import Any

import tiktoken

_TRUNC_SUFFIX = "\n...[truncated for context limit]"


@functools.lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += 4 + count_tokens(str(msg.get("content", "")))
    return total


def truncate_text(text: str, max_chars: int, *, suffix: str = _TRUNC_SUFFIX) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def truncate_content_to_tokens(content: str, max_tokens: int) -> str:
    """Binary-search truncate ``content`` so token count <= ``max_tokens``."""
    if max_tokens <= 0:
        return _TRUNC_SUFFIX
    if count_tokens(content) <= max_tokens:
        return content
    if count_tokens(_TRUNC_SUFFIX) > max_tokens:
        return content[: max(1, max_tokens * 2)]

    lo, hi = 0, len(content)
    best = _TRUNC_SUFFIX
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = content[:mid] + _TRUNC_SUFFIX
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fit_messages_to_budget(
    messages: list[dict[str, Any]],
    *,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Shrink message contents until the prompt fits ``max_prompt_tokens``."""
    if max_prompt_tokens <= 0:
        return [dict(m) for m in messages], False

    out = [dict(m) for m in messages]
    if count_messages_tokens(out) <= max_prompt_tokens:
        return out, False

    truncated = False
    for _ in range(64):
        if count_messages_tokens(out) <= max_prompt_tokens:
            break
        idx = max(range(len(out)), key=lambda i: len(str(out[i].get("content", ""))))
        content = str(out[idx].get("content", ""))
        other = count_messages_tokens(out) - count_tokens(content)
        allowed = max(32, max_prompt_tokens - other)
        new_content = truncate_content_to_tokens(content, allowed)
        if new_content == content:
            break
        out[idx]["content"] = new_content
        truncated = True

    return out, truncated
