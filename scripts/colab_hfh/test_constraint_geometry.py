import unittest
import numpy as np
import torch
from scripts.colab_hfh.constraint_geometry import DEPTH, sdf_targets, depth_weights


class GeometryTests(unittest.TestCase):
    def test_sdf_respects_thick_axis_and_sign(self):
        labels = torch.zeros(1, 1, 9, 9, 9, dtype=torch.long)
        labels[:, :, 4, 4, 4] = 1
        field = sdf_targets(labels)
        self.assertAlmostEqual(float(field[0, 0, 4, 4, 4]), -.5/8)
        self.assertAlmostEqual(float(field[0, 0, 5, 4, 4]), .5/8)
        self.assertAlmostEqual(float(field[0, 0, 4, 5, 4]), 1.5/8)
        self.assertLessEqual(float(field.abs().max()), 1.)

    def test_depth_normalized_and_rejects_wrong_spacing(self):
        labels = torch.zeros(1, 1, 64, 64, 64, dtype=torch.long)
        labels[:, :, 24:40, 28:36, 18:46] = 1
        affine = np.diag([1., 2., 1., 1.])
        weights = depth_weights(labels, affine, DEPTH)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertTrue(torch.isfinite(weights).all())
        with self.assertRaises(ValueError):
            depth_weights(labels, np.eye(4), DEPTH)


if __name__ == '__main__':
    unittest.main()
