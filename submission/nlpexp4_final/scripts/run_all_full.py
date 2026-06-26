"""一键跑完所有核心实验（5/26 中期检查交付主轴）。

按顺序触发：
  1. exp1 中文 main 全量 (N=300) × 5 ratio × 3 type
  2. exp1 中文 main 位置子实验（N=300, ratio=0.5）
  3. exp1 中文 fact 全量（counterfactual 专题）
  4. exp2 中文矫正方法对比 (N=300, methods=naive+4)
  5. exp3 中文 main 全量案例 (N=300, 全量分类, corrector=confidence)
  6. exp3 英文 main 全量案例 (N=300, 全量分类)
  7. exp3 中文 fact 全量案例 (N=100, 全量分类)
  8. exp3 英文 fact 全量案例 (N=100, 全量分类)
  9. exp1 英文 main 中等规模（N=100, 控制 token）
 10. exp2 英文矫正中等规模（N=50）

每个 step 单独可 enable/disable，默认全部开启。
跑前会做 token 用量预估 + 用户确认。
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
else:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Step:
    key: str
    desc: str
    cmd: list[str]
    enabled_by_default: bool = True
    # 预估每条样本的 LLM 调用次数（用于 token 成本提示）
    samples: int = 0
    calls_per_sample: int = 1
    avg_tokens_per_call: int = 2500  # prompt+completion 经验值


def _make_steps() -> list[Step]:
    return [
        Step(
            key="exp1_zh_main",
            desc="exp1 · 中文 main 主矩阵 (5 ratio × 3 type, N=300)",
            cmd=[
                sys.executable, "-m", "experiments.exp1_noise_impact",
                "--n", "300", "--language", "zh", "--subset", "main",
            ],
            samples=300, calls_per_sample=15,
        ),
        Step(
            key="exp1_zh_position",
            desc="exp1 · 中文 main 位置子实验 (ratio=0.5, 4 position, N=300)",
            cmd=[
                sys.executable, "-m", "experiments.exp1_noise_impact",
                "--n", "300", "--language", "zh", "--subset", "main",
                "--position-only",
            ],
            samples=300, calls_per_sample=4,
        ),
        Step(
            key="exp1_zh_fact",
            desc="exp1 · 中文 fact (counterfactual 专题, N=100)",
            cmd=[
                sys.executable, "-m", "experiments.exp1_noise_impact",
                "--n", "100", "--language", "zh", "--subset", "fact",
            ],
            samples=100, calls_per_sample=15,
        ),
        Step(
            key="exp2_zh",
            desc="exp2 · 中文矫正对比 (naive+prompt+iterative+confidence+selfrag, 4 ratio, N=300)",
            cmd=[
                sys.executable, "-m", "experiments.exp2_correction",
                "--n", "300", "--language", "zh",
            ],
            # naive=1, prompt=1, confidence=1, iterative≈11(10docs+1), selfrag≈12(10+2)
            samples=300, calls_per_sample=(1 + 1 + 1 + 11 + 12) * 4,
        ),
        Step(
            key="exp3_zh",
            desc="exp3 · 中文 main 全量案例 (N=300, 全量分类, corrector=confidence)",
            cmd=[
                sys.executable, "-m", "experiments.exp3_case_study",
                "--n", "300", "--pick", "30",
                "--language", "zh", "--corrector", "confidence",
            ],
            samples=300, calls_per_sample=7,  # 1 clean + 3 ratios × 2 methods
        ),
        Step(
            key="exp3_en",
            desc="exp3 · 英文 main 全量案例 (N=300, 全量分类, corrector=confidence)",
            cmd=[
                sys.executable, "-m", "experiments.exp3_case_study",
                "--n", "300", "--pick", "30",
                "--language", "en", "--corrector", "confidence",
            ],
            samples=300, calls_per_sample=7,
        ),
        Step(
            key="exp3_zh_fact",
            desc="exp3 · 中文 fact 全量案例 (N=100, 全量分类, corrector=confidence)",
            cmd=[
                sys.executable, "-m", "experiments.exp3_case_study",
                "--n", "100", "--pick", "30",
                "--language", "zh", "--subset", "fact",
                "--corrector", "confidence",
            ],
            samples=100, calls_per_sample=7,
        ),
        Step(
            key="exp3_en_fact",
            desc="exp3 · 英文 fact 全量案例 (N=100, 全量分类, corrector=confidence)",
            cmd=[
                sys.executable, "-m", "experiments.exp3_case_study",
                "--n", "100", "--pick", "30",
                "--language", "en", "--subset", "fact",
                "--corrector", "confidence",
            ],
            samples=100, calls_per_sample=7,
        ),
        Step(
            key="exp1_en_main",
            desc="exp1 · 英文 main 中等规模 (N=100)",
            cmd=[
                sys.executable, "-m", "experiments.exp1_noise_impact",
                "--n", "100", "--language", "en", "--subset", "main",
            ],
            samples=100, calls_per_sample=15,
        ),
        Step(
            key="exp2_en",
            desc="exp2 · 英文矫正对比 (中等规模, N=50)",
            cmd=[
                sys.executable, "-m", "experiments.exp2_correction",
                "--n", "50", "--language", "en",
                "--methods", "naive,prompt,confidence,selfrag",
            ],
            samples=50, calls_per_sample=(1 + 1 + 1 + 12) * 4,
        ),
    ]


def _estimate_cost(steps: list[Step]) -> dict:
    """粗略估算 token 用量。"""
    total_calls = 0
    total_tokens = 0
    rows = []
    for s in steps:
        calls = s.samples * s.calls_per_sample
        tokens = calls * s.avg_tokens_per_call
        total_calls += calls
        total_tokens += tokens
        rows.append((s.key, calls, tokens))
    # deepseek-chat ¥1 / 1M cached + ¥2 / 1M uncached output（粗估按 ¥2/M 算上限）
    cny_high = total_tokens / 1_000_000 * 2
    cny_low = total_tokens / 1_000_000 * 0.5  # 缓存命中后
    return {
        "rows": rows,
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "cny_high": cny_high,
        "cny_low": cny_low,
    }


def _print_plan(steps: list[Step], cost: dict) -> None:
    print("=" * 70)
    print("【实验执行计划】")
    print("=" * 70)
    for i, s in enumerate(steps, 1):
        print(f"  [{i}] {s.key:20s}  {s.desc}")
        print(f"      $ {' '.join(s.cmd[2:])}")
    print()
    print("【Token 成本预估（极粗略，仅供参考）】")
    print("-" * 70)
    for key, calls, tokens in cost["rows"]:
        print(f"  {key:20s}  {calls:>7,} calls   ~{tokens:>10,} tokens")
    print("-" * 70)
    print(
        f"  total                {cost['total_calls']:>7,} calls   "
        f"~{cost['total_tokens']:>10,} tokens"
    )
    print(
        f"  approx CNY {cost['cny_low']:.2f} (缓存命中较多) ~ "
        f"CNY {cost['cny_high']:.2f} (全部新调用)"
    )
    print()


def _run_step(step: Step, idx: int, total: int) -> tuple[bool, float]:
    print()
    print("#" * 70)
    print(f"#  [{idx}/{total}] {step.key}: {step.desc}")
    print(f"#  cmd: {' '.join(step.cmd)}")
    print("#" * 70)
    t0 = time.time()
    rc = subprocess.call(step.cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    ok = rc == 0
    tag = "OK" if ok else f"FAIL (rc={rc})"
    print(f"#  → {tag}  elapsed={elapsed:.1f}s")
    return ok, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    all_steps = _make_steps()
    all_keys = [s.key for s in all_steps]
    p.add_argument(
        "--only",
        default="",
        help=f"只跑指定 step（逗号分隔），可选: {','.join(all_keys)}",
    )
    p.add_argument(
        "--skip",
        default="",
        help="跳过指定 step（逗号分隔）",
    )
    p.add_argument("--yes", "-y", action="store_true", help="不交互确认，直接开跑")
    p.add_argument(
        "--dry-run", action="store_true", help="只打印计划与预估，不真跑"
    )
    args = p.parse_args()

    only = {k.strip() for k in args.only.split(",") if k.strip()}
    skip = {k.strip() for k in args.skip.split(",") if k.strip()}
    if only:
        unknown = only - set(all_keys)
        if unknown:
            print(f"未知 step: {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)
        steps = [s for s in all_steps if s.key in only]
    else:
        steps = [s for s in all_steps if s.key not in skip]

    if not steps:
        print("没有可执行的 step。", file=sys.stderr)
        sys.exit(2)

    cost = _estimate_cost(steps)
    _print_plan(steps, cost)

    if args.dry_run:
        print("(dry-run 模式，未真正执行)")
        return

    if not args.yes:
        try:
            ans = input("继续执行？输入 yes 确认: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in {"y", "yes"}:
            print("已取消。")
            return

    overall = []
    t0 = time.time()
    for i, s in enumerate(steps, 1):
        ok, elapsed = _run_step(s, i, len(steps))
        overall.append((s.key, ok, elapsed))

    total_elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("【全部执行完毕】")
    print("=" * 70)
    for key, ok, elapsed in overall:
        tag = "OK  " if ok else "FAIL"
        print(f"  {tag}  {key:20s}  {elapsed:>7.1f}s")
    print("-" * 70)
    print(f"  total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  results dir: {ROOT / 'experiments' / 'results'}")


if __name__ == "__main__":
    main()
