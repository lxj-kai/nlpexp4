"""NoiserBench / exp_dataset_correction 可视化单元测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.noiser_loader import NOISER_NOISE_TYPES
from src.visualize import (
    plot_noiser_method_heatmap,
    plot_noiser_type_compare,
    render_noiser_bench_figures,
)


def _fake_noiser_exp2_payload(*, ratio: float = 0.75) -> dict:
    results = []
    methods = ("naive", "prompt", "confidence")
    for method in methods:
        results.append(
            {
                "condition": {
                    "method": method,
                    "noise_ratio": 0.0,
                    "noise_type": "semantic",
                    "noise_position": "interleave",
                },
                "summary": {"judge_score": 0.8, "n": 10},
                "n": 10,
                "rows": [],
            }
        )
    for method in methods:
        for i, ntype in enumerate(NOISER_NOISE_TYPES):
            results.append(
                {
                    "condition": {
                        "method": method,
                        "noise_ratio": ratio,
                        "noise_type": ntype,
                        "noise_position": "interleave",
                    },
                    "summary": {"judge_score": max(0.1, 0.7 - i * 0.05), "n": 10},
                    "n": 10,
                    "rows": [],
                }
            )
    return {
        "experiment": "exp_noiser_exp2_hotpotqa_n10",
        "results": results,
        "robustness_table": [],
    }


@pytest.fixture()
def noiser_exp2_json(tmp_path: Path) -> Path:
    path = tmp_path / "exp_noiser_exp2_hotpotqa_n10_test.json"
    path.write_text(json.dumps(_fake_noiser_exp2_payload()), encoding="utf-8")
    return path


def test_plot_noiser_type_compare(noiser_exp2_json: Path, tmp_path: Path) -> None:
    out = plot_noiser_type_compare(noiser_exp2_json, out_dir=tmp_path)
    assert Path(out).exists()


def test_plot_noiser_method_heatmap(noiser_exp2_json: Path, tmp_path: Path) -> None:
    out = plot_noiser_method_heatmap(noiser_exp2_json, out_dir=tmp_path)
    assert Path(out).exists()


def test_render_noiser_bench_figures(noiser_exp2_json: Path, tmp_path: Path) -> None:
    paths = render_noiser_bench_figures(
        noiser_exp2_json,
        phase="exp2",
        out_dir=tmp_path,
    )
    assert len(paths) >= 2
    assert (tmp_path / "figures_manifest.json").exists()
