#!/usr/bin/env python3
"""Tests for epsilon-sweep visualization generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aggregate_epsilon_results import plot_epsilon_results


def summary(mean: float, deviation: float) -> dict[str, float | int]:
    return {"mean": mean, "std_population": deviation, "n": 5}


def sample_rows() -> list[dict[str, object]]:
    rows = []
    for grounding, offset in (("paper-hard", 0.0), ("soft-probability", 0.01)):
        for epsilon, index in ((0.0, 0), (500.0, 1), (5000.0, 2)):
            rows.append(
                {
                    "volume_grounding": grounding,
                    "volume_epsilon": epsilon,
                    "volume_gamma": 0.0001,
                    "train_fraction": 0.05,
                    "folds": [1, 2, 3, 4, 5],
                    "metrics": {
                        "dice_all_classes": summary(0.75 + offset + index * 0.005, 0.01),
                        "dice_foreground": summary(0.66 + offset + index * 0.008, 0.015),
                        "prediction_volume_gap_voxels": summary(1800 - index * 300, 120),
                        "prediction_volume_similarity": summary(1.0, 0.0),
                        "prediction_volume_similarity_configured": summary(0.2 + index * 0.4, 0.05),
                        "prediction_volume_violation_configured": summary(0.8 - index * 0.35, 0.04),
                    },
                }
            )
    return rows


class EpsilonVisualizationTests(unittest.TestCase):
    def test_all_expected_plots_are_valid_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_epsilon_results(sample_rows(), Path(temporary))
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
