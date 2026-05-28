"""One-click benchmark construction pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path

from config import BENCHMARK_ROOT


def main():
    print("=" * 60)
    print("  NLPExp4 Custom Benchmark Construction Pipeline")
    print("=" * 60)

    from crawl import main as crawl_main
    crawl_main()

    print()

    from generate import main as generate_main
    generate_main()

    print()

    from assemble import assemble
    records = assemble()

    output = BENCHMARK_ROOT / "output" / "bench_zh.json"
    target = BENCHMARK_ROOT.parent / "data" / "rgb" / "zh_bench.json"
    if output.exists():
        shutil.copy2(output, target)
        print(f"\nCopied to {target} for frontend access")

    print()
    print("=" * 60)
    print(f"  Done! {len(records)} benchmark entries created")
    print(f"  Output: {output}")
    print("=" * 60)
    print()
    print("Optional: Run quality_check.py for automated QA review")


if __name__ == "__main__":
    main()
