"""Verify error definitions and equal-volume aggregation independently."""
import unittest
from scripts.medsam3_multiclass_replication import confusion_metrics, average_rows


class ReplicationTests(unittest.TestCase):
    def test_error_categories_are_disjoint(self):
        row = confusion_metrics([[100, 3, 5], [7, 40, 11], [13, 17, 50]])
        self.assertEqual(row['foreground_fp'], 8)
        self.assertEqual(row['foreground_fn'], 20)
        self.assertEqual(row['a_to_p'], 11)
        self.assertEqual(row['p_to_a'], 17)
        self.assertEqual(row['ap_swaps'], 28)
        self.assertEqual(row['total_error_voxels'], 56)
        self.assertEqual(row['anterior_fp'], 20)
        self.assertEqual(row['anterior_fn'], 18)
        self.assertEqual(row['posterior_fp'], 16)
        self.assertEqual(row['posterior_fn'], 30)

    def test_averaging_is_macro_not_pooled(self):
        small = confusion_metrics([[100, 0, 0], [0, 1, 0], [0, 0, 1]])
        large = confusion_metrics([[100, 0, 0], [0, 0, 100], [0, 100, 0]])
        mean = average_rows([small, large])
        self.assertEqual(mean['ap_mean_dice'], .5)
        self.assertEqual(mean['ap_swaps'], 100)

    def test_background_does_not_inflate_primary_dice(self):
        row = confusion_metrics([[1000000, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(row['ap_mean_dice'], 0)
        self.assertGreater(row['background_dice'], .99)


if __name__ == '__main__':
    unittest.main()
