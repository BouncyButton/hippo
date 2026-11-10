"""Check error partitions and retrospective plateau definitions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostics", Path(__file__).with_name("analyze_unet_budget_diagnostics.py"))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticTests(unittest.TestCase):
    def label(self):
        label = np.zeros((24, 24, 24), dtype=np.uint8)
        label[4:12, 4:20, 4:20] = 1
        label[12:20, 4:20, 4:20] = 2
        return label

    def test_exact_error_partition_and_distances(self):
        label = self.label()
        pred = label.copy()
        pred[11, 10, 10] = 2  # internal interface, away from outer contour
        pred[8, 10, 10] = 2   # deep swap away from both two-voxel bands
        pred[4, 10, 10] = 0   # FN on outer contour
        pred[3, 10, 10] = 1   # FP immediately outside contour
        pred[0, 0, 0] = 1     # distant FP
        record = diagnostics.spatial_counts(label, pred)
        counts = record["counts"]
        self.assertEqual(counts["total_errors"], 5)
        self.assertEqual(counts["foreground_fp"], 2)
        self.assertEqual(counts["foreground_fn"], 1)
        self.assertEqual(counts["anterior_posterior_swap"], 2)
        self.assertEqual(counts["outer_only_errors"], 2)
        self.assertEqual(counts["interface_only_errors"], 1)
        self.assertEqual(counts["neither_band_errors"], 2)
        self.assertEqual(counts["all_outer_le_2"], 2)
        self.assertEqual(counts["swap_interface_le_2"], 1)
        self.assertEqual(counts["gt_deep_foreground_errors"], 2)
        self.assertGreater(record["dice_foreground_union"], (record["dice_anterior"] + record["dice_posterior"]) / 2)
        stats = diagnostics.aggregate_spatial([record, record])
        self.assertEqual(stats["counts"]["total_errors"], 10)
        self.assertAlmostEqual(stats["error_composition"]["foreground_fp"], 0.4)
        self.assertAlmostEqual(sum(stats["region_error_shares"].values()), 1)

    def test_perfect_prediction_and_overlapping_bands(self):
        label = self.label()
        perfect = diagnostics.spatial_counts(label, label)
        self.assertEqual(perfect["counts"]["total_errors"], 0)
        self.assertEqual(perfect["mean_symmetric_surface_distance_voxels"], 0)
        self.assertEqual(perfect["dice_foreground_union"], 1)
        pred = label.copy()
        pred[11, 4, 10] = 2
        counts = diagnostics.spatial_counts(label, pred)["counts"]
        self.assertEqual(counts["outer_and_interface_errors"], 1)
        self.assertEqual(counts["outer_only_errors"], 0)
        self.assertEqual(counts["interface_only_errors"], 0)

    def test_retrospective_tolerance_uses_best_so_far(self):
        curve = "step,macro_hard_dice,mean_train_loss\n10,0.5,0.5\n20,0.8,0.2\n30,0.79,0.1\n40,0.804,0.05\n"
        result = diagnostics.plateau_analysis([{"fold": f, "seed": s, "csv": curve} for f in range(3) for s in range(3)])
        self.assertEqual(result["thresholds"]["0.005"]["aggregate_first_step"], 20)
        self.assertAlmostEqual(result["milestones"][2]["mean_best_so_far_dice"], 0.8)
        self.assertAlmostEqual(result["milestones"][2]["mean_remaining_gain"], 0.004)
        self.assertEqual(result["thresholds"]["0.001"]["aggregate_first_step"], 40)


if __name__ == "__main__":
    unittest.main()
