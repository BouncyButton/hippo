"""Loss parity, exact restart, and full native-branch gradient checks."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.compute_constraint_objectives import training_loss as original_loss, calibration_case, select_weights as original_weights
from scripts.depth_supervision.losses import combined_loss
from scripts.depth_supervision.utils import atomic_json, seed_all
from scripts.depth_supervision.train import autocast, Patients
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.tissue_profiles.mri_views import MRIOnlyViewSwin, NativeViewAtlas, native_chart_inputs
from . import train
from .objectives import OBJECTIVES, BASE_OBJECTIVES, Targets, components, gradient_record, select_weights, terms


def synthetic_item(name='synthetic', size=16):
    generator = torch.Generator().manual_seed(91)
    labels = torch.zeros(1, size, size, size, dtype=torch.long)
    middle = size//2
    labels[:, middle-3:middle+3, middle-5:middle, middle-3:middle+3] = 1
    labels[:, middle-3:middle+3, middle:middle+5, middle-3:middle+3] = 2
    return dict(case_name=name, image=torch.randn(1, size, size, size, generator=generator),
                label=labels, affine=np.eye(4))


class TinyPatients:
    def __init__(self, dataframe, names, metadata):
        self.items = [synthetic_item(n) for n in names]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        return self.items[index]


class TinyBranch(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.native = torch.nn.Conv3d(1, 3, 1)
        self.encoder = torch.nn.Conv3d(1, 3, 1)
        self.fusion = torch.nn.Sequential(torch.nn.Dropout3d(.2), torch.nn.Conv3d(3, 3, 1))
        self.to(device)
    def forward(self, image, atlas):
        return self.native(image)+self.fusion(self.encoder(image))


class TinyAtlas:
    def get(self, item):
        return {}


class CampaignTests(unittest.TestCase):
    def test_exact_matrix(self):
        from .prepare import BASE_OBJECTIVES as declared
        self.assertEqual(BASE_OBJECTIVES, declared)
        self.assertEqual(len(set(OBJECTIVES)), 12)
        self.assertTrue(all('bands' in terms(o) for o in OBJECTIVES if 'edge' in terms(o)))

    def test_original_objectives_and_calibration(self):
        seed_all(0)
        item = synthetic_item()
        settings = dict(target_ratio=.1, cap_ratio=.5, prefix_epochs=5)
        with tempfile.TemporaryDirectory() as directory:
            target = Targets(Path(directory), dict(geometry={}), 'dice_bands_edge_sdf_depth').get(item, 'cpu')
            labels = item['label'][None]
            for objective in OBJECTIVES:
                z = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
                values = components(z, labels, objective, target)
                row = gradient_record(values, z)
                selected = select_weights([row], objective, settings)
                coefficients = selected['coefficients']
                base = objective.removesuffix('_depth')
                old = original_loss(z, labels, base, coefficients, .7, target['sdf'])
                if 'depth' in terms(objective):
                    old = old + .7*coefficients['depth']*values['depth']
                actual = combined_loss(values, coefficients, .7)
                torch.testing.assert_close(actual, old)
                torch.testing.assert_close(torch.autograd.grad(actual, z, retain_graph=True)[0],
                                           torch.autograd.grad(old, z)[0])
                expected, _ = original_weights([calibration_case(z.detach(), labels, base, target['sdf'])], base)
                for name, weight in expected.items():
                    self.assertAlmostEqual(coefficients[name], weight, places=5)
                self.assertTrue(all(v > 0 for v in coefficients.values()))
            self.assertAlmostEqual(float(target['weights'].sum()), 1., places=6)

    def test_restart_before_and_after_calibration(self):
        p = dict(identity_sha='test', splits={'5pct': dict(train=['a', 'b'], val=['v'])}, pkl='unused',
                 native_geometry={}, geometry={}, learning_rate=.001, weight_decay=1e-5,
                 step_size=3, lr_gamma=.5, calibration=dict(prefix_epochs=2, ramp_epochs=2,
                 target_ratio=.1, cap_ratio=.5, max_cases=10), max_epochs=5,
                 early_stopping=dict(patience=0, min_epochs=1, min_delta=0.))
        entry = dict(id='test', fraction='5pct', seed=0, arm='native_mri_context', objective='dice_bands_edge_sdf_depth')
        with tempfile.TemporaryDirectory() as directory, patch.object(train, '_load_pkl_dataframe', return_value=None), \
                patch.object(train, 'Patients', TinyPatients), \
                patch.object(train, 'model_and_inputs', side_effect=lambda r, p, f, d, a: (TinyBranch(d), TinyAtlas())):
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = train.run(p, root/'whole', entry, device_override='cpu')
                for stopped in (2, 3):
                    partial = root/str(stopped)
                    train.run(p, partial, entry, stop_after=stopped, device_override='cpu')
                    with self.assertRaisesRegex(ValueError, 'Resume identity'):
                        train.run(p, partial, dict(entry, seed=1), device_override='cpu')
                    resumed = train.run(p, partial, entry, device_override='cpu')
                    self.assertEqual(expected['validation'], resumed['validation'])
                    a = torch.load(root/'whole/runs/test/best.pt', weights_only=False)
                    b = torch.load(partial/'runs/test/best.pt', weights_only=False)
                    self.assertEqual(a['epoch'], b['epoch'])
                    for key in a['model']:
                        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
                    with (root/'whole/runs/test/metrics.csv').open() as fa, (partial/'runs/test/metrics.csv').open() as fb:
                        for left, right in zip(csv.DictReader(fa), csv.DictReader(fb)):
                            remove = ('elapsed_seconds', 'epoch_seconds')
                            self.assertEqual({k: v for k, v in left.items() if k not in remove},
                                             {k: v for k, v in right.items() if k not in remove})
                    self.assertEqual(train.run(p, partial, entry, device_override='cpu'), resumed)
                # Baseline remains exactly Dice-only and shares the initial prefix.
                baseline = dict(entry, id='dice', objective='dice')
                with patch.object(train, 'calibrate', side_effect=AssertionError('Baseline calibrated')):
                    train.run(p, root/'baseline', baseline, stop_after=2, device_override='cpu')
                with (root/'whole/runs/test/metrics.csv').open() as fa, (root/'baseline/runs/dice/metrics.csv').open() as fb:
                    for a, b in zip(csv.DictReader(fa), csv.DictReader(fb)):
                        self.assertEqual(a['train_dice'], b['train_dice'])
                        self.assertEqual(a['validation_dice'], b['validation_dice'])


def model_smoke(device_name, root=None):
    seed_all(0)
    device = torch.device(device_name)
    if root is not None:
        p = json.loads((root/'protocol.json').read_text())
        dataframe = _load_pkl_dataframe(p['pkl'])
        patients = Patients(dataframe, p['splits']['5pct']['train'], p['native_geometry'])
        del dataframe
        atlas_cache = NativeViewAtlas(root, p, '5pct')
        # Every training input and target must be valid before any experiment runs.
        targets = Targets(root, p, 'dice_bands_edge_sdf_depth')
        for item in patients.items:
            atlas_cache.get(item)
            targets.get(item, device)
            assert np.allclose(np.linalg.norm(item['affine'][:3, :3], axis=0), 1.)
        item = patients[0]
        atlas = atlas_cache.get(item)
        target = targets.get(item, device)
        cases_checked = len(patients)
    else:
        item = synthetic_item(size=64)
        roi = np.zeros((64, 64, 64), dtype=bool)
        roi[24:40, 20:44, 24:40] = True
        points = np.indices(roi.shape)[1]
        atlas = native_chart_inputs(item['image'][0].numpy(), [np.flatnonzero(roi & (points < 32)), np.flatnonzero(roi & (points >= 32))])
        with tempfile.TemporaryDirectory() as directory:
            target = Targets(Path(directory), dict(geometry={}), 'dice_bands_edge_sdf_depth').get(item, device)
        cases_checked = 0
    model = MRIOnlyViewSwin(device, 'native_mri_context')
    assert model.settings['view'] == 'native' and not model.settings['attention']
    assert all(x.requires_grad for x in model.parameters())
    assert all(model.encoder.get_submodule(name).in_channels == 1 for name in model.encoder.reduced_inputs)
    assert all(not key.startswith(('plane_', 'unfold_')) for key in atlas)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    with autocast(device):
        logits = model(item['image'][None].to(device), atlas)
    labels = item['label'][None].to(device)
    detached = logits.detach().float().requires_grad_(True)
    objective = 'dice_bands_edge_sdf_depth'
    selected = select_weights([gradient_record(components(detached, labels, objective, target), detached)],
                              objective, dict(target_ratio=.1, cap_ratio=.5, prefix_epochs=5))
    values = components(logits, labels, objective, target)
    loss = combined_loss(values, selected['coefficients'])
    loss.backward()
    groups = {}
    for name in ('native', 'encoder', 'fusion'):
        parameters = list(getattr(model, name).parameters())
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in parameters)
        groups[name] = float(sum(q.grad.abs().sum() for q in parameters))
        assert groups[name] > 0
    watched = {n: (q, q.detach().clone()) for n, q in model.named_parameters()
               if n in ('native.out.conv.conv.weight', 'encoder.fine.0.0.weight', 'fusion.0.0.weight')}
    assert len(watched) == 3
    optimizer.step()
    assert all(not torch.equal(q, before) for q, before in watched.values())
    return dict(passed=True, device=device_name, real_training_cases_checked=cases_checked,
                parameters=sum(q.numel() for q in model.parameters()), group_gradient_l1=groups,
                losses={k: float(v.detach()) for k, v in values.items()},
                coefficients=selected['coefficients'], native_mri_only=True, all_three_groups_updated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['unit', 'model'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'unit':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CampaignTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        value = dict(passed=True, tests=result.testsRun)
    else:
        value = model_smoke(args.device, args.root)
    source = Path(__file__).resolve().parents[2]
    from .prepare import sha
    paths = list((source/'scripts/native_branch_constraints').glob('*.py'))
    value['tested_sources'] = {str(path.relative_to(source)): sha(path) for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
