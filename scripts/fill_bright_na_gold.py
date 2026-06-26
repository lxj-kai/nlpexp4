"""Fill BRIGHT records whose source gold_answer is N/A via DeepSeek.

Source N/A rows come from code/math subdomains (leetcode, pony, aops, theoremqa).
Generates reference answers from query + positive docs, caches by subdomain:example_id,
then patches data/bright/en.json and en_fact.json.

Usage:
    python scripts/fill_bright_na_gold.py --limit 5          # smoke
    python scripts/fill_bright_na_gold.py                    # fill all missing
    python scripts/fill_bright_na_gold.py --apply-only       # patch JSONL from cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import is_usable_gold_answer  # noqa: E402
from src.llm_client import get_judge_client  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BRIGHT_DIR = ROOT / "data" / "bright"
CACHE_PATH = BRIGHT_DIR / "gold_fill_cache.json"

_CODE_SUBDOMAINS = frozenset({"leetcode", "pony"})
_MATH_SUBDOMAINS = frozenset({"aops", "theoremqa_questions", "theoremqa_theorems"})

FILL_SYSTEM = (
    "You write reference gold answers for a RAG benchmark. "
    "Use ONLY the provided question and reference documents. "
    "Be accurate and concise enough to grade against later.\n"
    "- Coding tasks: state the expected output or give a short correct solution sketch "
    "(key algorithm + result; code optional, max ~400 words).\n"
    "- Math tasks: give the final answer plus 2-5 sentences of justification.\n"
    "Output ONLY the reference answer. No preamble, no markdown headers."
)

FILL_USER_TMPL = (
    "Subdomain: {subdomain}\n\n"
    "Question:\n{query}\n\n"
    "Reference documents:\n{context}\n\n"
    "Reference gold answer:"
)


def fill_key(meta: dict) -> str:
    return f"{meta.get('subdomain', '?')}:{meta.get('example_id', '?')}"


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_context(docs: list[str], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, doc in enumerate(docs):
        chunk = doc[: max_chars - used]
        if not chunk:
            break
        parts.append(f"[Doc {i}]\n{chunk}")
        used += len(chunk)
        if used >= max_chars:
            break
    return "\n\n".join(parts) if parts else "(no documents)"


def _needs_fill(record: dict) -> bool:
    ans = record.get("answer")
    if isinstance(ans, list):
        raw = ans[0] if ans else ""
    else:
        raw = ans
    return not is_usable_gold_answer(raw)


def _collect_na_records(*paths: Path) -> dict[str, dict]:
    """Unique NA rows keyed by fill_key."""
    found: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not _needs_fill(rec):
                    continue
                meta = rec.get("meta") or {}
                key = fill_key(meta)
                found[key] = rec
    return found


def generate_gold(record: dict, *, llm) -> str:
    meta = record.get("meta") or {}
    subdomain = str(meta.get("subdomain", ""))
    query = str(record.get("query", "")).strip()
    positives = [str(x) for x in record.get("positive") or []]
    user = FILL_USER_TMPL.format(
        subdomain=subdomain,
        query=query,
        context=_format_context(positives),
    )
    out = llm.chat(
        [
            {"role": "system", "content": FILL_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    text = (out.get("content") or "").strip()
    if not text:
        raise RuntimeError("empty DeepSeek response")
    return text


def fill_missing(*, limit: int | None, dry_run: bool) -> dict[str, str]:
    cache = load_cache()
    pending = _collect_na_records(BRIGHT_DIR / "en.json", BRIGHT_DIR / "en_fact.json")
    todo = [k for k in pending if k not in cache or not is_usable_gold_answer(cache.get(k, ""))]
    if limit is not None:
        todo = todo[:limit]
    logger.info(f"NA records={len(pending)} cache={len(cache)} to_generate={len(todo)}")

    if dry_run or not todo:
        return cache

    llm = get_judge_client()
    for i, key in enumerate(todo, 1):
        rec = pending[key]
        logger.info(f"[{i}/{len(todo)}] generating {key}")
        try:
            cache[key] = generate_gold(rec, llm=llm)
            save_cache(cache)
        except Exception as e:
            logger.error(f"failed {key}: {e}")
    return cache


def apply_cache(cache: dict[str, str], path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    updated = 0
    total = 0
    lines_out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if _needs_fill(rec):
                key = fill_key(rec.get("meta") or {})
                filled = cache.get(key, "")
                if is_usable_gold_answer(filled):
                    if isinstance(rec.get("answer"), list):
                        rec["answer"] = [filled]
                    else:
                        rec["answer"] = filled
                    if rec.get("fakeanswer") in (None, "", "N/A", "NA"):
                        rec["fakeanswer"] = _flip_fact(filled)
                    rec.setdefault("meta", {})["gold_synthetic"] = True
                    updated += 1
            lines_out.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return updated, total


def _flip_fact(answer: str) -> str:
    low = answer.strip().lower()
    if low == "yes":
        return "no"
    if low == "no":
        return "yes"
    if len(answer) > 80:
        return answer[:77].rstrip() + "..."
    return f"not {answer}"


def main() -> None:
    p = argparse.ArgumentParser(description="Fill BRIGHT N/A gold answers via DeepSeek")
    p.add_argument("--limit", type=int, default=None, help="max new generations")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply-only", action="store_true", help="patch JSONL from cache only")
    args = p.parse_args()

    if args.apply_only:
        cache = load_cache()
    else:
        cache = fill_missing(limit=args.limit, dry_run=args.dry_run)

    if args.dry_run:
        print(json.dumps({"pending_cache_keys": len(cache)}, indent=2))
        return

    main_u, main_t = apply_cache(cache, BRIGHT_DIR / "en.json")
    fact_u, fact_t = apply_cache(cache, BRIGHT_DIR / "en_fact.json")
    save_cache(cache)
    print(
        json.dumps(
            {
                "cache_path": str(CACHE_PATH),
                "cache_size": len(cache),
                "en.json": {"updated": main_u, "total": main_t},
                "en_fact.json": {"updated": fact_u, "total": fact_t},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
