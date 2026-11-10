"""Checks for weight allocation, gradient direction, and exact resume behavior."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from monai.losses import DiceLoss

from .geometry import GeometryConfig, build_geometry
from .losses import ARMS, partial_class_weights, components, combined_loss, calibrated_coefficient
from .train import run


def phantom():
    y = np.zeros((20, 28, 20), np.uint8)
    y[7:13, 4:14, 6:14] = 1
    y[7:13, 14:24, 6:14] = 2
    return y


class WeightTests(unittest.TestCase):
    def test_band_mass_roi_and_boost_cap(self):
        y = phantom()
        g = build_geometry(y, np.eye(4), fields=False)
        w = partial_class_weights(y, g)
        self.assertTrue(np.isfinite(w).all())
        self.assertTrue((w[g['weights'] == 0] == 0).all())
        self.assertAlmostEqual(float(w.sum()), 1., places=6)
        self.assertGreater(float(w[y > 0].sum()), float(g['weights'][y > 0].sum()))
        roi = g['weights'] > 0
        self.assertLessEqual(float((w[roi] / g['weights'][roi]).max()), 4.00001)
        for b in g['bin_values']:
            m = g['bin_ids'] == b
            self.assertAlmostEqual(float(w[m].sum()), float(g['weights'][m].sum()), places=6)

    def test_single_class_and_balanced_classes_remain_unchanged(self):
        for y in [np.zeros((3, 10, 10), np.uint8), np.repeat(np.arange(3), 100).reshape(3, 10, 10)]:
            g = dict(weights=np.full(y.shape, 1/y.size, np.float32), bin_ids=np.zeros(y.shape, np.int16))
            np.testing.assert_allclose(partial_class_weights(y, g), g['weights'], rtol=1e-6)

    def test_tiny_class_cannot_dominate(self):
        y = np.zeros((10, 10, 10), np.uint8)
        y[0, 0, 0] = 1
        g = dict(weights=np.full(y.shape, .001, np.float32), bin_ids=np.zeros(y.shape, np.int16))
        w = partial_class_weights(y, g)
        self.assertLessEqual(float(w[0, 0, 0]), .004001)
        self.assertGreater(float(w[y == 0].sum()), .99)


class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = phantom()
        cls.g = build_geometry(cls.y, np.eye(4), fields=False)
        cls.g['class_weights'] = partial_class_weights(cls.y, cls.g)
        cls.target = torch.from_numpy(cls.y.astype(np.int64))[None, None]

    def test_exact_dice_warmup(self):
        z = torch.randn(1, 3, *self.y.shape, requires_grad=True)
        value = components(z, self.target, arm='dice')
        self.assertEqual(list(value), ['dice'])
        torch.testing.assert_close(combined_loss(value, {}), DiceLoss(to_onehot_y=True, softmax=True)(z, self.target), rtol=0, atol=0)

    def test_all_six_confusions_have_correct_gradient_direction(self):
        points = {0:(6,9,10), 1:(9,9,10), 2:(9,19,10)}
        for arm in ARMS:
            for true in range(3):
                for wrong in range(3):
                    if true == wrong:
                        continue
                    z = torch.zeros(1, 3, *self.y.shape, requires_grad=True)
                    point = points[true]
                    with torch.no_grad():
                        z[(0,wrong,*point)] = 5.
                    v = components(z, self.target, self.g, arm)
                    self.assertEqual(set(v), {'dice', 'depth'})
                    grad = torch.autograd.grad(v['depth'], z)[0]
                    self.assertLess(float(grad[(0,true,*point)] - grad[(0,wrong,*point)]), 0.)
                    self.assertEqual(float(grad[0, :, self.g['weights'] == 0].abs().sum()), 0.)

    def test_focal_suppresses_easy_errors_and_has_finite_extremes(self):
        def gradient(correct_logit):
            z = torch.zeros(1, 3, 1, 1, 1, requires_grad=True)
            with torch.no_grad():
                z[0,1] = correct_logit
            y = torch.ones(1,1,1,1,1,dtype=torch.long)
            v = components(z, y, {'weights':np.ones((1,1,1),np.float32)}, 'depth_focal')['depth']
            g = torch.autograd.grad(v,z)[0]
            self.assertTrue(torch.isfinite(v) and torch.isfinite(g).all())
            return float(g.abs().sum())
        self.assertLess(gradient(5.), gradient(-5.) * .01)
        self.assertEqual(gradient(100.), 0.)
        self.assertGreater(gradient(-100.), 0.)

    def test_finite_difference_both_objectives(self):
        torch.manual_seed(0)
        for arm in ARMS:
            z = torch.randn(1,3,*self.y.shape,requires_grad=True)
            location=(0,1,9,9,10)
            v=components(z,self.target,self.g,arm)['depth']
            analytical=torch.autograd.grad(v,z)[0][location].item()
            plus,minus=z.detach().clone(),z.detach().clone()
            plus[location]+=.05
            minus[location]-=.05
            numerical=(components(plus,self.target,self.g,arm)['depth']-components(minus,self.target,self.g,arm)['depth']).item()/.1
            self.assertAlmostEqual(analytical,numerical,delta=2e-6)

    def test_calibration_cap(self):
        result=calibrated_coefficient([1]*10,[1]*9+[100])
        self.assertLessEqual(result['weighted_ratio_p95'],.5000001)


class TinyPatients:
    def __init__(self, name):
        y=phantom()
        image=y.astype(np.float32)+np.random.default_rng(0).normal(0,.1,y.shape).astype(np.float32)
        self.items=[dict(case_name=name,image=torch.from_numpy(image)[None],label=torch.from_numpy(y.astype(np.int64))[None],affine=np.eye(4))]
    def __len__(self):
        return len(self.items)
    def __getitem__(self,i):
        return self.items[i]


class TrainingTests(unittest.TestCase):
    def test_resume_and_train_only_calibration_for_both_arms(self):
        protocol=dict(seed=0,max_epochs=3,learning_rate=.001,weight_decay=1e-5,step_size=2,lr_gamma=.5,
            geometry=vars(GeometryConfig()),early_stopping=dict(patience=8,min_epochs=25,min_delta=.0005),
            calibration=dict(prefix_epochs=1,ramp_epochs=1,target_ratio=.1,cap_ratio=.5),
            splits={'test':{'train':['train'],'val':['validation']}})
        patients=(TinyPatients('train'),TinyPatients('validation'))
        factory=lambda device:torch.nn.Conv3d(1,3,1).to(device)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            for arm in ARMS:
                common=dict(protocol=protocol,fraction='test',arm=arm,pkl=None,cache_dir=root/'cache',device='cpu',test_model_factory=factory,test_patients=patients)
                one=run(output=root/arm/'one',**common)
                run(output=root/arm/'two',stop_after_epoch=2,**common)
                two=run(output=root/arm/'two',**common)
                self.assertEqual(one['best_epoch'],two['best_epoch'])
                self.assertEqual(one['validation'],two['validation'])
                self.assertEqual(one['coefficients'],two['coefficients'])
                a=torch.load(root/arm/'one/checkpoint_latest.pt',map_location='cpu',weights_only=False)
                b=torch.load(root/arm/'two/checkpoint_latest.pt',map_location='cpu',weights_only=False)
                self.assertEqual([r['case_name'] for r in a['calibration']['cases']],['train'])
                self.assertEqual(a['updates'],b['updates'])
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
                self.assertEqual(run(output=root/arm/'two',**common)['artifacts'],two['artifacts'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
