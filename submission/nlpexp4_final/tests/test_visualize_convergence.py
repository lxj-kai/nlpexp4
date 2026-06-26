"""Tests for iterative convergence plotting."""
from __future__ import annotations

import pytest

from src.visualize import plot_iterative_convergence


class TestPlotIterativeConvergence:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="round_logs"):
            plot_iterative_convergence([])

    def test_from_round_logs(self, tmp_path):
        logs = [
            [
                {"round": 0, "isr": 0.4, "nar": 0.3},
                {"round": 1, "isr": 0.5, "nar": 0.2},
            ],
            [
                {"round": 0, "isr": 0.6, "nar": 0.1},
                {"round": 1, "isr": 0.7, "nar": 0.05},
            ],
        ]
        out = plot_iterative_convergence(logs, out_dir=tmp_path)
        assert out.endswith("iterative_convergence.png")
