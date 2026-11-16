import unittest
import numpy as np
from scripts.colab_hfh.constraint_calibration import select_joint_weights, extra_calibration_flops


class JointCalibrationTests(unittest.TestCase):
    def test_joint_cap_matches_direct_gradient_sum(self):
        vectors = [np.array([[1., 0.], [3., 1.], [2., -1.]]),
                   np.array([[.5, .2], [2., 1.], [1., -2.]])]
        rows = [dict(gradient_names=['native','bands','edge'], gradient_gram=(v@v.T/2).tolist()) for v in vectors]
        weights, stats = select_joint_weights(rows,'dice_bands_edge',targets={'bands':.6,'edge':.6},
                                               component_cap=1.5,joint_target=.3,joint_cap=.4)
        ratios = [np.linalg.norm(weights['bands']*v[1]+weights['edge']*v[2])/np.linalg.norm(v[0]) for v in vectors]
        self.assertAlmostEqual(float(np.median(ratios)),stats['joint']['weighted_median_ratio'])
        self.assertAlmostEqual(float(np.quantile(ratios,.95)),stats['joint']['weighted_p95_ratio'])
        self.assertLessEqual(np.median(ratios),.3+1e-12)
        self.assertLessEqual(np.quantile(ratios,.95),.4+1e-12)
        self.assertLess(stats['joint']['shrink'],1.)

    def test_antagonistic_gradients_are_reported_not_silently_flipped(self):
        vectors=np.array([[1.,0.],[-2.,.1]])
        rows=[dict(gradient_names=['native','sdf'],gradient_gram=(vectors@vectors.T).tolist())]
        weights,stats=select_joint_weights(rows,'dice_sdf',targets={'sdf':.1},component_cap=.5,joint_target=.1,joint_cap=.5)
        self.assertGreater(weights['sdf'],0)
        self.assertEqual(stats['joint']['alignment']['native:sdf']['negative_cases'],1)
        self.assertLess(stats['joint']['alignment']['native:sdf']['median_cosine'],0)
        self.assertGreater(extra_calibration_flops('dice_bands_edge_sdf_depth'),extra_calibration_flops('dice_bands'))

    def test_delayed_term_cannot_hide_a_large_early_gradient(self):
        vectors = np.array([[1., 0.], [10., 1.], [-9., -1.]])
        rows = [dict(gradient_names=['native','bands','sdf'], gradient_gram=(vectors@vectors.T).tolist())]
        weights, stats = select_joint_weights(rows, 'dice_bands_sdf', targets={'bands': 1., 'sdf': 1.},
            component_cap=2., joint_target=.1, joint_cap=.2)
        for sdf_scale in np.linspace(0., 1., 11):
            ratio = np.linalg.norm(weights['bands']*vectors[1]+sdf_scale*weights['sdf']*vectors[2])
            self.assertLessEqual(ratio, .1+1e-12)
        self.assertLessEqual(stats['joint']['worst_ramp_median_ratio'], .1+1e-12)


if __name__=='__main__':unittest.main()
