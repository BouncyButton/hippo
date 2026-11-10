"""Geometry, supervision direction, calibration, and training/resume checks."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from monai.losses import DiceLoss

from .geometry import GeometryConfig, build_geometry, longitudinal_plane, signed_distance
from .losses import ARMS, calibrated_coefficient, combined_loss, components
from .train import run
from .utils import stopping_state


def phantom(shape=(20, 28, 20)):
    y = np.zeros(shape, np.uint8)
    y[7:13, 4:14, 6:14] = 1
    y[7:13, 14:24, 6:14] = 2
    return y


class GeometryTests(unittest.TestCase):
    def test_plane_and_physical_equivariance(self):
        y = phantom()
        p = longitudinal_plane(y, np.eye(4))
        np.testing.assert_allclose(p['normal'], [1, 0, 0], atol=1e-7)
        self.assertAlmostEqual(float(p['normal'] @ p['basis'][:, 0]), 0.)
        angle = .41
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        affine = np.eye(4)
        affine[:3, :3] = r
        affine[:3, 3] = [13, -7, 5]
        q = longitudinal_plane(y, affine, right_axis=r[:, 0], anterior_axis=r[:, 1])
        np.testing.assert_allclose(q['normal'], r @ p['normal'], atol=1e-7)
        np.testing.assert_allclose(q['center_mm'], r @ p['center_mm'] + affine[:3, 3], atol=1e-7)

    def test_depth_mass_signs_and_background(self):
        y = phantom()
        g = build_geometry(y, np.eye(4))
        self.assertAlmostEqual(float(g['weights'].sum()), 1., places=6)
        self.assertTrue((g['weights'][y == 0] > 0).any())
        self.assertTrue((g['bin_values'] < 0).any() and (g['bin_values'] >= 0).any())
        for b, mass in zip(g['bin_values'], g['bin_masses']):
            self.assertAlmostEqual(float(g['weights'][g['bin_ids'] == b].sum()), float(mass), places=6)
        for field in ('sdf_3d', 'sdf_layer'):
            for c in (1, 2):
                self.assertTrue((g[field][c-1][y == c] < 0).all())
                self.assertTrue((g[field][c-1][y != c] > 0).all())

    def test_layer_distance_removes_cross_depth_shortcut(self):
        y = phantom()
        g = build_geometry(y, np.eye(4))
        # A voxel near the thin LR surface but far from the within-layer contour.
        point = (7, 9, 10)
        self.assertAlmostEqual(abs(float(g['sdf_3d'][(0, *point)])) * 8, .5)
        self.assertGreater(abs(float(g['sdf_layer'][(0, *point)])), abs(float(g['sdf_3d'][(0, *point)])))

    def test_existing_3d_convention_and_empty_sections(self):
        y = phantom()
        from scipy.ndimage import distance_transform_edt
        mask = y == 1
        expected = np.where(mask, -(distance_transform_edt(mask)-.5), distance_transform_edt(~mask)-.5)
        np.testing.assert_allclose(signed_distance(mask, (1, 1, 1), 8), np.clip(expected, -8, 8))
        self.assertTrue((signed_distance(np.zeros((10, 10), bool), (1, 1), 8) == 8).all())

    def test_oblique_shapes_preserve_native_signs(self):
        x, y, z = np.indices((24, 30, 24))
        mask = ((x - 12 - .18*(y-15))/4)**2 + ((y-15)/11)**2 + ((z-12)/5)**2 < 1
        labels = np.where(mask, np.where(y < 15, 1, 2), 0).astype(np.uint8)
        g = build_geometry(labels, np.eye(4))
        for c in (1, 2):
            self.assertTrue((g['sdf_layer'][c-1][labels == c] < 0).all())
            self.assertTrue((g['sdf_layer'][c-1][labels != c] > 0).all())
        self.assertLess(abs(float(np.dot(g['normal'], g['basis'][:, 0]))), 1e-6)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = phantom()
        cls.g = build_geometry(cls.y, np.eye(4))
        cls.target = torch.from_numpy(cls.y.astype(np.int64))[None, None]

    def test_dice_only_is_exact_baseline(self):
        logits = torch.randn(1, 3, *self.y.shape, requires_grad=True)
        actual = components(logits, self.target, arm='dice')
        expected = DiceLoss(to_onehot_y=True, softmax=True)(logits, self.target)
        self.assertEqual(list(actual), ['dice'])
        torch.testing.assert_close(actual['dice'], expected, rtol=0, atol=0)
        torch.testing.assert_close(combined_loss(actual, {}, 1), expected, rtol=0, atol=0)

    def test_directed_errors_depth_gradient_and_sdf_probability_transfer(self):
        points = {0: (6, 9, 10), 1: (9, 9, 10), 2: (9, 19, 10)}
        for arm in ARMS[1:]:
            for true in (0, 1, 2):
                for wrong in (0, 1, 2):
                    if true == wrong:
                        continue
                    logits = torch.zeros(1, 3, *self.y.shape, requires_grad=True)
                    point = points[true]
                    with torch.no_grad():
                        logits[(0, wrong, *point)] = 3.
                    values = components(logits, self.target, self.g, arm)
                    for name in values:
                        if name == 'dice':
                            continue
                        grad = torch.autograd.grad(values[name], logits, retain_graph=True)[0]
                        if name == 'depth':
                            self.assertLess(float(grad[(0, true, *point)] - grad[(0, wrong, *point)]), 0.)
                        else:
                            # SDF is linear in probabilities. Transferring mass
                            # directly from the wrong to true class must help.
                            # A softmax gradient can also redistribute mass to
                            # the third class, so universal logit-margin descent
                            # is NOT a property of the existing classwise SDF.
                            field = self.g['sdf_3d' if arm == 'depth_sdf3d' else 'sdf_layer']
                            costs = [0., float(field[(0, *point)]), float(field[(1, *point)])]
                            self.assertLess(costs[true] - costs[wrong], 0.)

    def test_finite_difference_and_saturation(self):
        logits = torch.randn(1, 3, *self.y.shape, requires_grad=True)
        location = (0, 1, 9, 9, 10)
        loss = components(logits, self.target, self.g, 'depth_sdf_layer')['sdf']
        analytic = torch.autograd.grad(loss, logits)[0][location].item()
        delta = .02
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[location] += delta
        minus[location] -= delta
        numerical = (components(plus, self.target, self.g, 'depth_sdf_layer')['sdf'] -
                     components(minus, self.target, self.g, 'depth_sdf_layer')['sdf']).item() / (2*delta)
        self.assertAlmostEqual(analytic, numerical, delta=2e-6)
        perfect = torch.full_like(logits, -25.)
        perfect.scatter_(1, self.target, 25.)
        self.assertLess(abs(float(components(perfect, self.target, self.g, 'depth_sdf_layer')['sdf'])), 1e-8)

    def test_calibration_cap(self):
        result = calibrated_coefficient([1]*10, [1]*9+[100])
        self.assertLessEqual(result['weighted_ratio_p95'], .5000001)
        with self.assertRaises(ValueError):
            calibrated_coefficient([1], [0])


class TinyPatients:
    def __init__(self, name):
        y = phantom((20, 28, 20))
        image = torch.from_numpy((y.astype(np.float32) + np.random.default_rng(0).normal(0, .1, y.shape)).astype(np.float32))[None]
        self.items = [dict(case_name=name, image=image, label=torch.from_numpy(y.astype(np.int64))[None], affine=np.eye(4))]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


class TrainingTests(unittest.TestCase):
    def test_early_stopping_accumulation(self):
        history = [dict(epoch=i+1, validation_dice=s) for i, s in enumerate([.8, .8003, .8006, .8007, .8007])]
        state = stopping_state(history, patience=2, min_epochs=5, min_delta=.0005)
        self.assertEqual(state['reference_epoch'], 3)
        self.assertTrue(state['triggered'])

    def test_resume_matches_uninterrupted_updates_and_selection(self):
        protocol = dict(seed=0, max_epochs=3, learning_rate=.001, weight_decay=1e-5,
                        step_size=2, lr_gamma=.5, geometry=vars(GeometryConfig()),
                        early_stopping=dict(patience=8, min_epochs=25, min_delta=.0005),
                        calibration=dict(prefix_epochs=1, ramp_epochs=1, target_ratio=.1, cap_ratio=.5),
                        splits={'test': {'train': ['train'], 'val': ['validation']}})
        patients = (TinyPatients('train'), TinyPatients('validation'))
        factory = lambda device: torch.nn.Conv3d(1, 3, 1).to(device)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(protocol=protocol, fraction='test', arm='depth_sdf_layer', pkl=None,
                          cache_dir=root/'cache', device='cpu', test_model_factory=factory, test_patients=patients)
            one = run(output=root/'one', **common)
            run(output=root/'two', stop_after_epoch=2, **common)
            two = run(output=root/'two', **common)
            self.assertEqual(one['best_epoch'], two['best_epoch'])
            self.assertEqual(one['coefficients'], two['coefficients'])
            a = torch.load(root/'one/checkpoint_latest.pt', map_location='cpu', weights_only=False)
            b = torch.load(root/'two/checkpoint_latest.pt', map_location='cpu', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['updates'], b['updates'])
            self.assertEqual(one['validation'], two['validation'])
            self.assertEqual(run(output=root/'two', **common)['artifacts'], two['artifacts'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
