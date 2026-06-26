"""Tests for batch run visualization helpers."""
from __future__ import annotations

import json
from pathlib import Path

from src.visualize import (
    _detect_score_key,
    _summary_score,
    render_batch_run_figures,
)


def _mini_payload() -> dict:
    def block(method: str, ratio: float, score: float):
        return {
            "condition": {
                "method": method,
                "noise_ratio": ratio,
                "noise_type": "semantic",
                "noise_position": "interleave",
            },
            "summary": {
                "n": 5,
                "judge_score": score,
                "isr": 0.8 - ratio * 0.3,
                "nar": ratio * 0.4,
            },
            "rows": [],
            "elapsed": 1.0,
            "n": 5,
        }

    return {
        "experiment": "smoke_test",
        "results": [
            block("naive", 0.0, 0.9),
            block("naive", 0.5, 0.6),
            block("prompt", 0.0, 0.88),
            block("prompt", 0.5, 0.72),
        ],
        "robustness_table": [
            {
                "method": "naive",
                "noise_type": "semantic",
                "NS": 0.33,
                "NRS": -0.6,
                "CRR": None,
                "ISR_avg": 0.65,
                "NAR_avg": 0.2,
            },
            {
                "method": "prompt",
                "noise_type": "semantic",
                "NS": 0.18,
                "NRS": -0.32,
                "CRR": 0.2,
                "ISR_avg": 0.7,
                "NAR_avg": 0.18,
            },
        ],
    }


def test_summary_score_prefers_judge():
    assert _summary_score({"judge_score": 0.8, "token_f1": 0.3}) == 0.8


def test_detect_score_key():
    payload = _mini_payload()
    key, label = _detect_score_key(payload)
    assert key == "judge_score"
    assert label == "Judge Score"


def test_render_batch_run_figures(tmp_path: Path):
    jf = tmp_path / "smoke_mini.json"
    jf.write_text(json.dumps(_mini_payload()), encoding="utf-8")
    out_dir = tmp_path / "figs"
    paths = render_batch_run_figures(jf, out_dir=out_dir, tag="mini")
    assert len(paths) >= 4
    for p in paths:
        assert Path(p).exists()
