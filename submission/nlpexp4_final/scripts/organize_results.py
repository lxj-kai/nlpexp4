"""将 experiments/results、figures、logs 按实验类型归档到子目录。

用法：
    python scripts/organize_results.py              # 预览
    python scripts/organize_results.py --apply      # 执行移动
    python scripts/organize_results.py --apply --write-index
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

# (regex on stem/filename, category path under each root)
RESULT_RULES: list[tuple[str, str]] = [
    (r"^exp1_noise_impact", "midterm/exp1_noise_impact"),
    (r"^exp2_correction", "midterm/exp2_correction"),
    (r"^exp3_case_study", "midterm/exp3_case_study"),
    (r"^exp4_existing_methods", "midterm/exp4_existing_methods"),
    (r"^exp5_deep", "midterm/exp5_deep"),
    (r"^exp_2wiki_", "dataset_2wiki"),
    (r"^exp_new_exp1", "dataset_new/exp1_noise_impact"),
    (r"^exp_new_exp2", "dataset_new/exp2_correction"),
    (r"^exp_correction_", "dataset_new/exp2_correction"),
    (r"^exp_noiser_", "dataset_noiser"),
    (r"^exp_noise_gradient", "exp_noise_gradient"),
    (r"^exp_closed_book", "exp_closed_book"),
    (r"^sanity_", "sanity"),
    (r"^smoke_", "smoke"),
    (r"^dryrun", "misc"),
]

FIGURE_RULES: list[tuple[str, str]] = [
    (r"^smoke_", "smoke"),
    (r"^exp_correction_", "dataset_new"),
    (r"^exp_noiser_", "dataset_noiser"),
    (r"^exp1_", "midterm/exp1"),
    (r"^exp2_", "midterm/exp2"),
    (r"^2wiki", "dataset_2wiki"),
]

LOG_RULES: list[tuple[str, str]] = [
    (r"^exp_2wiki_", "dataset_2wiki"),
    (r"^exp_noise_gradient", "exp_noise_gradient"),
    (r"^exp_n50_", "exp_noise_gradient"),
    (r"^run_2wiki_", "dataset_2wiki"),
    (r"^run_tempo_multihop_noiser", "dataset_noiser"),
    (r"^exp_noiser_", "dataset_noiser"),
    (r"^exp_correction_", "dataset_new"),
    (r"^smoke_", "smoke"),
    (r"^sanity_", "sanity"),
    (r"^fill_bright", "data_prep"),
    (r"^prepare_", "data_prep"),
    (r"^sample_judge", "misc"),
]


def classify(name: str, rules: list[tuple[str, str]]) -> str:
    for pat, cat in rules:
        if re.search(pat, name):
            return cat
    return "misc"


def _plan_moves(
    root: Path,
    items: list[Path],
    rules: list[tuple[str, str]],
    *,
    is_dir: bool = False,
) -> list[tuple[Path, Path]]:
    plans: list[tuple[Path, Path]] = []
    for src in items:
        if not src.exists():
            continue
        if is_dir:
            key = src.name
        else:
            key = src.stem
        cat = classify(key, rules)
        if is_dir:
            dest = root / cat / src.name
        else:
            dest = root / cat / src.name
        if src.resolve() == dest.resolve():
            continue
        if dest.exists():
            # 同名已存在则跳过（避免覆盖）
            continue
        plans.append((src, dest))
    return plans


def collect_flat_json() -> list[Path]:
    if not RESULTS.exists():
        return []
    return sorted(p for p in RESULTS.glob("*.json") if p.is_file())


def collect_figure_dirs() -> list[Path]:
    if not FIGURES.exists():
        return []
    return sorted(p for p in FIGURES.iterdir() if p.is_dir())


def collect_flat_logs() -> list[Path]:
    if not LOGS.exists():
        return []
    return sorted(p for p in LOGS.glob("*.log") if p.is_file())


def apply_moves(plans: list[tuple[Path, Path]], *, dry_run: bool) -> int:
    n = 0
    for src, dest in plans:
        if dry_run:
            print(f"  [dry] {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            print(f"  moved {src.name} -> {dest.parent.relative_to(ROOT)}/")
        n += 1
    return n


def write_index(manifest: dict) -> None:
    idx_json = RESULTS / "INDEX.json"
    idx_md = RESULTS / "README.md"
    idx_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 实验结果目录说明",
        "",
        f"归档时间：{manifest['organized_at']}",
        "",
        "## 目录结构",
        "",
        "```text",
        "experiments/results/",
        "├── midterm/              # 中期 RGB 主实验 exp1–exp5",
        "├── dataset_2wiki/        # 2Wiki 专项",
        "├── dataset_new/          # Cmedqa / MIRIAD 新数据集",
        "├── exp_noise_gradient/   # 三数据集噪音梯度",
        "├── exp_closed_book/      # 闭卷基线",
        "├── sanity/               # 新数据集 sanity check",
        "├── smoke/                # 批量 smoke 评测",
        "└── misc/                 # 其它",
        "```",
        "",
        "## 各目录文件数",
        "",
    ]
    for cat, files in sorted(manifest["categories"].items()):
        lines.append(f"- `{cat}/`：{len(files)} 个 JSON")
    lines.extend(["", "完整清单见 `INDEX.json`。", ""])
    idx_md.write_text("\n".join(lines), encoding="utf-8")


def build_manifest() -> dict:
    categories: dict[str, list[str]] = defaultdict(list)
    for path in sorted(RESULTS.rglob("*.json")):
        if path.name in ("INDEX.json",):
            continue
        rel = path.relative_to(RESULTS)
        cat = str(rel.parent) if rel.parent != Path(".") else "root"
        categories[cat].append(str(rel))
    return {
        "organized_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "total_json": sum(len(v) for v in categories.values()),
        "categories": dict(sorted(categories.items())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="归档实验结果 JSON / figures / logs")
    p.add_argument("--apply", action="store_true", help="执行移动（默认仅预览）")
    p.add_argument("--write-index", action="store_true", help="写入 INDEX.json 与 README.md")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-logs", action="store_true")
    args = p.parse_args()
    dry = not args.apply

    print("=== experiments/results (*.json) ===")
    r_plans = _plan_moves(RESULTS, collect_flat_json(), RESULT_RULES)
    print(f"  {len(r_plans)} files to classify")
    apply_moves(r_plans, dry_run=dry)

    if not args.skip_figures:
        print("\n=== figures/ (subdirs) ===")
        f_plans = _plan_moves(FIGURES, collect_figure_dirs(), FIGURE_RULES, is_dir=True)
        print(f"  {len(f_plans)} dirs to classify")
        apply_moves(f_plans, dry_run=dry)

    if not args.skip_logs:
        print("\n=== logs/*.log ===")
        l_plans = _plan_moves(LOGS, collect_flat_logs(), LOG_RULES)
        print(f"  {len(l_plans)} logs to classify")
        apply_moves(l_plans, dry_run=dry)

    if args.write_index or args.apply:
        manifest = build_manifest()
        if not dry:
            write_index(manifest)
            print(f"\nindex -> {RESULTS / 'README.md'}")
        else:
            print(f"\n[dry] would index {manifest['total_json']} json files in subdirs")

    if dry:
        print("\n预览完成。执行归档：python scripts/organize_results.py --apply --write-index")


if __name__ == "__main__":
    main()
