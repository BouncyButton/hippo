"""Geometry, loss/gradient parity, resume, and real-data model preflight checks."""
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
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, autocast, model_factory
from scripts.depth_supervision.losses import combined_loss
from scripts.depth_supervision.utils import atomic_json, digest, seed_all
from scripts.native_branch_constraints.objectives import gradient_record, select_weights
from scripts.native_branch_constraints.objectives import components as original_components
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from .objectives import (Targets, components, crossing_loss, edge_value, experiment_matrix,
                         make_pairs, pair_weights, valid_field)
from . import train


def synthetic_item(name='synthetic', size=16):
    generator = torch.Generator().manual_seed(91)
    labels = torch.zeros(1, size, size, size, dtype=torch.long)
    m = size//2
    labels[:, m-3:m+3, m-5:m, m-3:m+3] = 1
    labels[:, m-3:m+3, m:m+5, m-3:m+3] = 2
    return dict(case_name=name, image=torch.randn(1, size, size, size, generator=generator),
                label=labels, affine=np.eye(4))


def target_for(item, entry):
    p = dict(native_geometry={item['case_name']: dict(native_shape=list(item['label'].shape[1:]))})
    return Targets(Path('.'), p, entry).get(item, 'cpu')


class TinyPatients:
    def __init__(self, dataframe, names, metadata):
        self.items = [synthetic_item(n) for n in names]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


def tiny_model(device):
    return torch.nn.Sequential(torch.nn.Conv3d(1, 6, 1), torch.nn.Dropout3d(.2),
                               torch.nn.Conv3d(6, 3, 1)).to(device)


class CampaignTests(unittest.TestCase):
    def test_matrix(self):
        entries = experiment_matrix()
        self.assertEqual(len(entries), 20)
        self.assertEqual(len({e['id'] for e in entries}), 20)
        self.assertEqual(sum(e['kind'] == 'cross' for e in entries), 16)

    def test_original_loss_and_gradients(self):
        seed_all(0)
        item = synthetic_item()
        y = item['label'][None]
        z = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
        for entry in experiment_matrix()[:2]:
            value = components(z, y, entry, target_for(item, entry))
            old = original_components(z, y, 'dice_bands' if entry['kind'] == 'bands' else 'dice_bands_edge', {})
            for k in value:
                torch.testing.assert_close(value[k], old[k], atol=0, rtol=0)
                torch.testing.assert_close(torch.autograd.grad(value[k], z, retain_graph=True)[0],
                                           torch.autograd.grad(old[k], z, retain_graph=True)[0], atol=0, rtol=0)
        target = target_for(item, dict(kind='same_only'))
        same = edge_value(z, y, dict(kind='same_only'), target)
        cross = crossing_loss(z, target['legacy_cross'], 'outer', 'probability')
        n, m = len(target['same']['i']), len(target['legacy_cross']['i'])
        recovered = (same*n+cross*m)/(n+m)
        legacy = BoundaryEdgeConsistencyLoss()(z, y)
        torch.testing.assert_close(recovered, legacy)
        torch.testing.assert_close(torch.autograd.grad(recovered, z, retain_graph=True)[0],
                                   torch.autograd.grad(legacy, z)[0], atol=1e-8, rtol=1e-5)

    def test_geometry_single_transition_and_physical_distance(self):
        item = synthetic_item()
        y = item['label'][0].numpy()
        valid = valid_field(y.shape, (14, 14, 14))
        for interface in ('outer', 'ap'):
            pair = make_pairs(y, valid, [1., 1., 1.], interface, (1., 3.), 'anchors')
            self.assertGreater(pair['stats']['by_distance']['3.0'], 0)
            self.assertAlmostEqual(float(pair['weight'].sum()), 1., places=6)
            for i, j, scale in zip(pair['i'], pair['j'], pair['scale']):
                a, b = np.array(np.unravel_index(i, y.shape)), np.array(np.unravel_index(j, y.shape))
                steps = (1, 3)[scale]
                self.assertEqual(int(np.abs(b-a).sum()), steps)
                direction = (b-a)//steps
                for k in range(steps+1):
                    voxel = tuple(a+k*direction)
                    self.assertTrue(valid[voxel])
                    inside = k <= steps//2
                    if interface == 'outer':
                        self.assertEqual(bool(y[voxel] > 0), inside)
                    else:
                        self.assertEqual(int(y[voxel]), 1 if inside else 2)

    def test_no_padding_and_thin_structure_fallback(self):
        y = np.zeros((9, 9, 9), np.uint8)
        y[4, 2:7, 2:7] = 1
        valid = np.zeros_like(y, bool)
        valid[4:8, 2:7, 2:7] = True
        pair = make_pairs(y, valid, [1, 1, 1], 'outer', (1., 3.), 'anchors')
        self.assertEqual(pair['stats']['pairs'], 25)
        self.assertEqual(pair['stats']['by_distance']['3.0'], 0)
        self.assertTrue(np.all(valid.ravel()[pair['i']]) and np.all(valid.ravel()[pair['j']]))
        torch.testing.assert_close(pair['weight'], torch.full((25,), 1/25))
        # Adjacent A/P labels are required; a background gap is not an interface.
        y[6, 2:7, 2:7] = 2
        self.assertEqual(make_pairs(y, valid, [1, 1, 1], 'ap', (1., 3.))['stats']['pairs'], 0)

    def test_anchor_and_scale_normalization(self):
        anchors = np.array([1, 1, 1, 2, 2, 2, 2])
        scales = np.array([0, 0, 1, 0, 0, 0, 0])
        weights = pair_weights(anchors, scales, 'anchors')
        self.assertAlmostEqual(weights[anchors == 1].sum(), .5)
        self.assertAlmostEqual(weights[anchors == 2].sum(), .5)
        self.assertAlmostEqual(weights[(anchors == 1) & (scales == 0)].sum(), .25)
        weights = pair_weights(anchors, scales, 'pairs')
        self.assertAlmostEqual(weights[scales == 0].sum(), .5)
        self.assertAlmostEqual(weights[scales == 1].sum(), .5)

    def test_empty_regions_are_finite_zero(self):
        for label in (0, 1, 2):
            y = np.full((5, 5, 5), label, np.uint8)
            for interface in ('outer', 'ap'):
                pair = make_pairs(y, np.ones_like(y, bool), [1, 1, 1], interface, (1., 3.))
                z = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
                for form in ('probability', 'score'):
                    value = crossing_loss(z, pair, interface, form)
                    self.assertEqual(value.item(), 0)
                    self.assertTrue((torch.autograd.grad(value, z, retain_graph=True)[0] == 0).all())

    def test_score_keeps_gradient_when_confidently_wrong(self):
        pair = dict(i=torch.tensor([0]), j=torch.tensor([1]), weight=torch.ones(1))
        z = torch.tensor([[-40., 40.], [0., 0.], [0., 0.]]).reshape(1, 3, 1, 1, 2).requires_grad_()
        # Here foreground score is HIGH inside and LOW outside: correctly ordered.
        good = crossing_loss(z, pair, 'outer', 'score')
        wrong = z.detach().flip(-1).requires_grad_()
        value = crossing_loss(wrong, pair, 'outer', 'score')
        self.assertGreater(value.item(), good.item())
        grad = torch.autograd.grad(value, wrong)[0]
        self.assertGreater(grad[0, 0, 0, 0, 0].item(), .9)  # lower inside background logit
        self.assertLess(grad[0, 0, 0, 0, 1].item(), -.9)  # raise outside background logit
        self.assertTrue(torch.isfinite(grad).all())

    def test_ap_swap_symmetry(self):
        item = synthetic_item()
        y = item['label'][0].numpy()
        swap = np.where(y == 1, 2, np.where(y == 2, 1, 0))
        z = torch.randn(1, 3, *y.shape, requires_grad=True)
        a = make_pairs(y, np.ones_like(y, bool), [1, 1, 1], 'ap')
        b = make_pairs(swap, np.ones_like(y, bool), [1, 1, 1], 'ap')
        for form in ('probability', 'score'):
            torch.testing.assert_close(crossing_loss(z, a, 'ap', form),
                                       crossing_loss(z[:, [0, 2, 1]], b, 'ap', form))

    def test_exact_restart_and_matched_prefix(self):
        p = dict(identity_sha='test', splits={'5pct': dict(train=['a', 'b'], val=['v'])}, pkl='unused',
                 native_geometry={n: dict(native_shape=[16]*3) for n in ('a', 'b', 'v')}, geometry={},
                 learning_rate=.001, weight_decay=1e-5, step_size=3, lr_gamma=.5,
                 calibration=dict(prefix_epochs=2, ramp_epochs=2, target_ratio=.1, cap_ratio=.5, max_cases=10),
                 max_epochs=5, early_stopping=dict(patience=0, min_epochs=1, min_delta=0.))
        entry = dict(experiment_matrix()[-1], id='test', fraction='5pct', seed=0, arm='ordinary_swinunetr')
        with tempfile.TemporaryDirectory() as directory, patch.object(train, '_load_pkl_dataframe', return_value=None), \
                patch.object(train, 'Patients', TinyPatients), patch.object(train, 'model_factory', side_effect=tiny_model):
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
                baseline = dict(entry, id='bands', kind='bands')
                base_record = train.run(p, root/'whole', baseline, device_override='cpu')
                p['runs'] = [baseline, entry]
                from .report import report
                result = report(p, root/'whole')
                self.assertEqual(result['completed'], 2)
                self.assertEqual(len(result['paired_deltas']['test_minus_bands']), 1)
                with self.assertRaisesRegex(ValueError, 'Completed run identity'):
                    train.run(p, root/'whole', dict(baseline, seed=2), device_override='cpu')


def model_smoke(device_name, root, pkl_override=None):
    seed_all(0)
    device = torch.device(device_name)
    p = json.loads((root/'protocol.json').read_text())
    pkl = pkl_override or p['pkl']
    if digest(pkl) != p['pkl_sha256']:
        raise ValueError('Smoke-test dataset hash mismatch')
    dataframe = _load_pkl_dataframe(pkl)
    patients = Patients(dataframe, p['splits']['5pct']['train'], p['native_geometry'])
    del dataframe
    targets, geometry = {}, {}
    for entry in p['runs']:
        cache = Targets(root, p, entry)
        for item in patients.items:
            target = cache.get(item, 'cpu')
            if not target['outer']['i'].numel() or not target['ap']['i'].numel():
                raise ValueError('Real training case lacks required outer/AP crossings')
        targets[entry['id']] = cache
        geometry[entry['id']] = cache.statistics()
    item = patients[0]
    model = model_factory(device)
    assert sum(q.numel() for q in model.parameters()) == p['model_parameters']
    assert all(q.requires_grad for q in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    with autocast(device):
        logits = model(item['image'][None].to(device))
    labels = item['label'][None].to(device)
    detached = logits.detach().float().requires_grad_(True)
    checks = {}
    for entry in p['runs']:
        target = targets[entry['id']].get(item, device)
        values = components(detached, labels, entry, target)
        record = gradient_record(values, detached)
        weights = select_weights([record], 'dice_bands' if entry['kind'] == 'bands' else 'dice_bands_edge', p['calibration'])
        checks[entry['id']] = dict(losses=record['losses'], coefficients=weights['coefficients'])
    entry = p['runs'][-1]
    values = components(logits, labels, entry, targets[entry['id']].get(item, device))
    loss = combined_loss(values, checks[entry['id']]['coefficients'])
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    watched = {n: (q, q.detach().clone()) for n, q in model.named_parameters()
               if n in ('swinViT.patch_embed.proj.weight', 'out.conv.conv.weight')}
    assert len(watched) == 2
    optimizer.step()
    assert all(not torch.equal(q, before) for q, before in watched.values())
    return dict(passed=True, device=device_name, real_training_cases_checked=len(patients),
                parameters=sum(q.numel() for q in model.parameters()), objectives_checked=len(checks),
                objective_checks=checks, geometry=geometry, ordinary_swinunetr=True, parameters_updated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['unit', 'model'], required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--pkl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'unit':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CampaignTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        value = dict(passed=True, tests=result.testsRun)
    else:
        value = model_smoke(args.device, args.root, args.pkl)
    from .prepare import ROOT, sha
    paths = list((ROOT/'scripts/edge_ablation').glob('*.py')) + list((ROOT/'scripts/edge_ablation').glob('*.sh'))
    value['tested_sources'] = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('tested_sources', 'geometry', 'objective_checks')}))


if __name__ == '__main__':
    main()
