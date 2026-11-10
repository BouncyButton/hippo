"""Analytical geometry/gradient checks and real-data SwinUNETR smoke checks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.edge_ablation.objectives import Targets as OldTargets, components as old_components
from scripts.depth_supervision.utils import atomic_json, digest, seed_all
from scripts.depth_supervision.losses import combined_loss
from scripts.native_branch_constraints.objectives import gradient_record
from .geometry import Targets, edge_grid, surface_points, pair_geometry
from .matrix import experiment_matrix
from .objectives import components, point_chamfer, surface_loss, band_loss, select_weights
from . import train


def target_for(item, entry):
    p = dict(native_geometry={item['case_name']: dict(native_shape=list(item['label'].shape[1:]))})
    return Targets(Path('.'), p, entry).get(item, 'cpu')


class ScientificTests(unittest.TestCase):
    def test_matrix_and_fixed_controls(self):
        entries = experiment_matrix()
        self.assertEqual(len(entries), 64)
        self.assertEqual({family: sum(e['family'] == family for e in entries) for family in {e['family'] for e in entries}},
                         dict(control=2, band=13, pair=10, surface=13, combination=7,
                              ap_control=3, ap_chamfer=9, ap_distribution=3, ap_combination=4))

    def test_ap_reversal_needs_semantic_guard_and_bg_gradient_is_zero(self):
        from .ap import ap_target, ap_loss, ap_ce_loss
        y = np.zeros((9, 11, 9), np.int64)
        y[2:7, 2:6, 2:7] = 2; y[2:7, 6:9, 2:7] = 1
        target = ap_target(y, np.ones_like(y, bool), np.ones(3))
        score = torch.tensor(np.where(y == 1, 6., -6.), dtype=torch.float32)
        z = torch.stack((torch.zeros_like(score), score/2, -score/2))[None].requires_grad_()
        spec = dict(kind='chamfer', penalty='squared')
        good = ap_loss(z, spec, target)
        swapped = z[:, [0, 2, 1]]
        torch.testing.assert_close(good, ap_loss(swapped, spec, target))
        labels = torch.tensor(y)[None, None]
        ce_good = ap_ce_loss(z, labels, dict(kind='full'), target)
        ce_bad = ap_ce_loss(swapped, labels, dict(kind='full'), target)
        self.assertGreater(float(ce_bad), float(ce_good)+5)
        gradient = torch.autograd.grad(ce_bad+good, z)[0]
        self.assertTrue((gradient[:, 0] == 0).all())

    def test_ap_displacement_mixed_slice_and_distribution_gradients(self):
        from .ap import ap_target, ap_loss
        y = np.zeros((9, 13, 9), np.int64)
        y[2:7, 2:7, 2:7] = 2; y[2:7, 7:11, 2:7] = 1
        target = ap_target(y, np.ones_like(y, bool), np.ones(3))
        def logits(cut):
            score = torch.tensor(np.broadcast_to(3*(np.arange(13)[None, :, None]-(cut-.5)), y.shape).copy(), dtype=torch.float32)
            return torch.stack((torch.zeros_like(score), score/2, -score/2))[None].requires_grad_()
        for kind in ('chamfer', 'column_chamfer', 'cut_w1', 'cut_expected_l1', 'sdf'):
            spec = dict(kind=kind, temperature=.1, clip_mm=8)
            exact = ap_loss(logits(7), spec, target)
            for cut in (5, 9):
                z = logits(cut)
                displaced = ap_loss(z, spec, target)
                self.assertGreater(float(displaced), float(exact))
                gradient = torch.autograd.grad(displaced, z)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.)
                self.assertTrue((gradient[:, 0] == 0).all())
        y[2:4, 7, 2:7] = 2
        mixed = ap_target(y, np.ones_like(y, bool), np.ones(3))
        self.assertGreater(int((mixed['target_distribution'] > 0).sum()), 1)
        self.assertAlmostEqual(float(mixed['target_distribution'].sum()), 1., places=6)
        self.assertGreater(len(mixed['gt']), 0)

    def test_control_losses_and_gradients_match_previous(self):
        item = synthetic_item()
        labels = item['label'][None]
        logits = torch.randn(1, 3, *labels.shape[2:], requires_grad=True)
        protocol = dict(native_geometry={'synthetic': dict(native_shape=list(labels.shape[2:]))})
        for entry in experiment_matrix()[:2]:
            old_entry = dict(kind='bands') if entry['id'] == 'bands' else dict(
                kind='cross', form='probability', distances_mm=[1., 3.], reduction='pairs', ap=False)
            old = old_components(logits, labels, old_entry, OldTargets(Path('.'), protocol, old_entry).get(item, 'cpu'))
            new = components(logits, labels, entry, target_for(item, entry))
            for key in new:
                expected = old['edge' if key == 'pair' else key]
                torch.testing.assert_close(new[key], expected, atol=1e-7, rtol=1e-6)
                torch.testing.assert_close(torch.autograd.grad(new[key], logits, retain_graph=True)[0],
                                           torch.autograd.grad(expected, logits, retain_graph=True)[0], atol=1e-8, rtol=1e-5)

    def test_interpolated_coordinates_and_gradient(self):
        grid = edge_grid(np.ones((2, 1, 1), bool), np.ones(3))
        probability = torch.tensor([.2, .7], dtype=torch.double, requires_grad=True).reshape(2, 1, 1)
        actual = surface_points(probability, grid)
        torch.testing.assert_close(actual, torch.tensor([[.6, 0., 0.]], dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(lambda p: surface_points(p, grid), (probability,), eps=1e-6, atol=1e-5))

    def test_chamfer_values_and_gradients_both_directions(self):
        pred = torch.tensor([[.3, .2, 0.], [2.4, .1, .2]], dtype=torch.double, requires_grad=True)
        truth = torch.tensor([[0., 0., 0.], [2., 0., 0.], [3., .4, .1]], dtype=torch.double)
        for power in ('linear', 'squared', 'huber'):
            spec = dict(penalty=power, predicted_weight=.5)
            self.assertTrue(torch.autograd.gradcheck(lambda x: point_chamfer(x, truth, spec), (pred,), eps=1e-6, atol=1e-5))
        distances = torch.cdist(pred, truth)
        expected = .5*(distances.min(1).values.mean()+distances.min(0).values.mean())
        torch.testing.assert_close(point_chamfer(pred, truth, {}), expected)
        torch.testing.assert_close(point_chamfer(pred, truth, {}), point_chamfer(truth, pred, {}))
        torch.testing.assert_close(point_chamfer(pred, truth, dict(predicted_weight=1.)), distances.min(1).values.mean())

    def test_chamfer_gradient_moves_boundary_towards_truth(self):
        grid = edge_grid(np.ones((2, 1, 1), bool), np.ones(3))
        p = torch.tensor([.2, .7], dtype=torch.double, requires_grad=True).reshape(2, 1, 1)
        truth = torch.tensor([[.4, 0., 0.]], dtype=torch.double)
        def objective(value):
            return point_chamfer(surface_points(value, grid), truth, dict(penalty='squared'))
        self.assertTrue(torch.autograd.gradcheck(objective, (p,), eps=1e-6, atol=1e-5))
        loss = objective(p)
        gradient = torch.autograd.grad(loss, p)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertLess(float(objective(p-.01*gradient)), float(loss))

    def test_perfect_surface_empty_surfaces_and_padding(self):
        item = synthetic_item()
        entry = next(e for e in experiment_matrix() if e['id'] == 'chamfer_mean')
        target = target_for(item, entry)
        p = (item['label'][0] > 0).float().requires_grad_()
        loss = surface_loss(p, entry['surface'], target)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, p)[0]).all())
        for value in (0., 1.):
            p = torch.full_like(p, value, requires_grad=True)
            loss = surface_loss(p, entry['surface'], target)
            self.assertEqual(float(loss), 0.)
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss, p)[0]).all())
        self.assertEqual(target['runtime']['empty_prediction_calls'], 2)
        for value in (0, 1):
            empty = synthetic_item()
            empty['label'].fill_(value)
            target = target_for(empty, entry)
            p = torch.rand_like(p, requires_grad=True)
            self.assertEqual(float(surface_loss(p, entry['surface'], target)), 0.)
            self.assertEqual(target['runtime']['empty_target_calls'], 1)
        valid = np.zeros((6, 6, 6), bool); valid[1:5, 1:5, 1:5] = True
        grid = edge_grid(valid, np.array([2., 1., 1.]))
        self.assertTrue(valid.reshape(-1)[grid['i']].all() and valid.reshape(-1)[grid['j']].all())
        p = torch.tensor(valid, dtype=torch.float32)
        self.assertEqual(len(surface_points(p, grid)), 0)  # padding is not a boundary

    def test_common_support_and_shift_geometry(self):
        item = synthetic_item()
        y = item['label'][0].numpy()
        valid = np.ones_like(y, bool)
        entries = {e['id']: e for e in experiment_matrix()}
        counts = []
        for key in ('1', '3', '13_equal', '13_far75'):
            spec = entries['pair_common_'+key]['pair']
            pair = pair_geometry(y, valid, np.ones(3), spec)
            counts.append(len(pair['i'])/len(spec['distances']))
            for k, (distance, weight) in enumerate(zip(spec['distances'], spec['weights'])):
                use = pair['scale'] == k
                a, b = [np.stack(np.unravel_index(pair[q][use], y.shape), 1) for q in ('i', 'j')]
                self.assertTrue((np.abs(b-a).sum(1) == distance).all())
                self.assertAlmostEqual(float(pair['weight'][use].sum()), weight, places=6)
        self.assertEqual(len(set(counts)), 1)
        anchors = []
        for shift in ('symmetric', 'outward', 'inward'):
            pair = pair_geometry(y, valid, np.ones(3), entries['pair_shift_'+shift]['pair'])
            use = pair['scale'] == 0
            anchors.append(pair['i'][use])
            self.assertTrue((y.reshape(-1)[pair['i']] > 0).all())
            self.assertTrue((y.reshape(-1)[pair['j']] == 0).all())
        torch.testing.assert_close(anchors[0], anchors[1]); torch.testing.assert_close(anchors[0], anchors[2])

    def test_physical_shells_have_local_and_correct_gradient_signs(self):
        item = synthetic_item()
        entry = next(e for e in experiment_matrix() if e['id'] == 'band_euclid1')
        target = target_for(item, entry)
        logits = torch.zeros(1, 3, *item['label'].shape[1:], requires_grad=True)
        loss = band_loss(logits, item['label'][None], entry['band'], target)
        grad = torch.autograd.grad(loss, logits)[0][0, 0]
        support = target['valid'] & (target['distance'] <= 1)
        self.assertTrue((grad[~support] == 0).all())
        self.assertTrue((grad[support & target['foreground']] > 0).all())
        self.assertTrue((grad[support & ~target['foreground']] < 0).all())

    def test_joint_calibration_budget(self):
        entry = next(e for e in experiment_matrix() if e['id'] == 'pair13_plus_mean')
        item = synthetic_item()
        logits = torch.randn(1, 3, *item['label'].shape[1:], requires_grad=True)
        values = components(logits, item['label'][None], entry, target_for(item, entry))
        row = gradient_record(values, logits)
        result = select_weights([row], entry, dict(prefix_epochs=5, target_ratio=.1, cap_ratio=.5))
        weights = result['coefficients']
        base = values['dice']+weights['bands']*values['bands']
        extra = sum(weights[k]*values[k] for k in ('pair', 'surface'))
        a = torch.autograd.grad(extra, logits, retain_graph=True)[0]
        b = torch.autograd.grad(base, logits)[0]
        self.assertAlmostEqual(float(a.square().mean().sqrt()/b.square().mean().sqrt()), .1, places=5)

    def test_restart_does_not_change_training(self):
        p = dict(identity_sha='test', splits={'5pct': dict(train=['a', 'b'], val=['v'])}, pkl='unused',
                 native_geometry={n: dict(native_shape=[16]*3) for n in ('a', 'b', 'v')}, geometry={},
                 learning_rate=.001, weight_decay=1e-5, step_size=3, lr_gamma=.5,
                 calibration=dict(prefix_epochs=2, ramp_epochs=2, target_ratio=.1, cap_ratio=.5, max_cases=10),
                 max_epochs=4, early_stopping=dict(patience=0, min_epochs=1, min_delta=0.))
        entry = next(e for e in experiment_matrix() if e['id'] == 'pair13_plus_sdf8')
        with tempfile.TemporaryDirectory() as directory, patch.object(train, '_load_pkl_dataframe', return_value=None), \
                patch.object(train, 'Patients', TinyPatients), patch.object(train, 'model_factory', side_effect=tiny_model):
            root = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = train.run(p, root/'whole', entry, device_override='cpu')
                train.run(p, root/'resume', entry, stop_after=3, device_override='cpu')
                actual = train.run(p, root/'resume', entry, device_override='cpu')
            self.assertEqual(expected['validation'], actual['validation'])
            a = torch.load(root/'whole/runs'/entry['id']/'best.pt', weights_only=False)
            b = torch.load(root/'resume/runs'/entry['id']/'best.pt', weights_only=False)
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], atol=0, rtol=0)


def model_smoke(device_name, root, pkl_override=None):
    seed_all(0)
    device = torch.device(device_name)
    p = json.loads((root/'protocol.json').read_text())
    dataset = pkl_override or p['pkl']
    if digest(dataset) != p['pkl_sha256']:
        # Local pickle serialization differs. Permit only the independently
        # cluster-verified ten image/label array fingerprints, never guess parity.
        from .audit_ap import audit
        actual, _ = audit(p, dataset)
        expected = json.loads((root/'ap_training_geometry_cluster.json').read_text())
        if actual != expected:
            raise ValueError('Training arrays differ from the frozen cluster dataset')
    patients = train.Patients(train._load_pkl_dataframe(dataset), p['splits']['5pct']['train'], p['native_geometry'])
    model = train.model_factory(device)
    prefix = root/'shared_warmup/latest.pt'
    if prefix.exists():
        state = torch.load(prefix, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); del state
    assert sum(q.numel() for q in model.parameters()) == p['model_parameters']
    item = patients[0]
    model.eval()
    with train.autocast(device):
        logits = model(item['image'][None].to(device))
    detached = logits.detach().float().requires_grad_(True)
    labels = item['label'][None].to(device)
    checks, geometry = {}, {}
    for entry in p['runs']:
        cache = Targets(root, p, entry)
        for other in patients.items:
            target = cache.get(other, 'cpu')
            if not len(target['gt']):
                raise ValueError('No GT surface in real training case')
            if entry['pair'] and not len(target['pair']['i']):
                raise ValueError('No pair support in real training case')
        values = components(detached, labels, entry, cache.get(item, device))
        record = gradient_record(values, detached)
        calibrated = select_weights([record], entry, p['calibration'])
        checks[entry['id']] = dict(losses=record['losses'], coefficients=calibrated['coefficients'])
        geometry[entry['id']] = cache.statistics()
        print(json.dumps(dict(stage='smoke_objective', id=entry['id'], losses=record['losses'])), flush=True)
    entry = next(e for e in p['runs'] if e['id'] == 'pair13_plus_mean')
    values = components(logits, labels, entry, Targets(root, p, entry).get(item, device))
    surface_gradient = torch.autograd.grad(values['surface'], logits, retain_graph=True)[0]
    assert float(surface_gradient.abs().sum()) > 0
    loss = combined_loss(values, checks[entry['id']]['coefficients'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    watched = {n: (q, q.detach().clone()) for n, q in model.named_parameters()
               if n in ('swinViT.patch_embed.proj.weight', 'out.conv.conv.weight')}
    assert len(watched) == 2
    optimizer.step()
    assert all(not torch.equal(q, before) for q, before in watched.values())
    return dict(passed=True, objectives_checked=len(checks), real_training_cases_checked=len(patients),
                device=device_name, parameters=p['model_parameters'], parameters_updated=True,
                ordinary_swinunetr=True, chamfer_logit_gradient_nonzero=True,
                objective_checks=checks, geometry=geometry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('unit', 'model'), required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pkl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'unit':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        value = dict(passed=True, tests=result.testsRun)
    else:
        value = model_smoke(args.device, args.root, args.pkl)
    from .prepare import ROOT, sha
    paths = [f for f in (ROOT/'scripts/band_chamfer').glob('*') if f.suffix in ('.py', '.sh')]
    value['tested_sources'] = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('tested_sources', 'geometry', 'objective_checks')}))


if __name__ == '__main__':
    main()
