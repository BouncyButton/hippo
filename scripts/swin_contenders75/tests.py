"""Check original-loss equivalence, requested scope and real GPU gradients."""
import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.depth_supervision.utils import atomic_json, seed_all, digest
from scripts.depth_supervision.train import Patients, model_factory, autocast
from scripts.depth_supervision.losses import combined_loss
from scripts.native_branch_constraints import objectives as original
from scripts.band_focus import combo_objectives
from scripts.profile_unet_constraint_budget import synthetic_target
from scripts.capped_winner.tests import GEOMETRY
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from . import objectives


class Tests(unittest.TestCase):
    def test_archive_verification_precedes_deletion(self):
        from . import worker
        entry = objectives.entries()[2]
        p = dict(identity_sha='test', splits={'5pct':dict(train=[], val=[])})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root/'runs'/entry['id']
            dest.mkdir(parents=True)
            names = ('best.pt', 'validation_predictions.npz', 'metrics.csv')
            for name in names:
                (dest/name).write_bytes(name.encode())
            (dest/'unrelated.pt').write_bytes(b'preserve')
            record = dict(identity=worker.train.identity(p, entry),
                          artifacts={name:digest(dest/name) for name in names})
            atomic_json(dest/'completion.json', record)
            with patch.object(worker.shutil, 'copyfile', side_effect=OSError('archive unavailable')):
                with self.assertRaises(OSError):
                    worker.archive_and_retire(root, p, entry)
            self.assertTrue((dest/'best.pt').exists())
            worker.archive_and_retire(root, p, entry)
            self.assertFalse((dest/'best.pt').exists())
            self.assertFalse((dest/'validation_predictions.npz').exists())
            self.assertTrue((dest/'unrelated.pt').exists())
            self.assertEqual(digest(root/'recorded_results'/entry['id']/'metrics.csv'),
                             record['artifacts']['metrics.csv'])
            worker.archive_and_retire(root, p, entry)

    def test_scope_and_original_loss_equivalence(self):
        torch.set_num_threads(2)
        seed_all(0)
        entries = objectives.entries()
        self.assertEqual(len(entries), 15)
        self.assertEqual(len({e['id'] for e in entries}), 15)
        self.assertEqual({e['id'] for e in entries[2:]}, {name for name, _ in objectives.ROWS[1:]})
        self.assertTrue(all(e['seed'] == e['fold'] == 0 and e['fraction'] == '5pct' for e in entries))
        labels = synthetic_target('ellipsoid')
        target = dict(sdf=original.signed_distance_maps(labels, clip=8.),
                      weights=torch.ones((64,)*3)/(64**3))
        logits = torch.randn(1, 3, 64, 64, 64, requires_grad=True)
        settings = dict(prefix_epochs=5, target_ratio=.1, cap_ratio=.5)
        full = original.components(logits, labels, 'dice_bands_edge_sdf_depth', target)
        for entry in entries:
            if entry['id'] == 'winning_constraint':
                continue
            actual = objectives.components(logits, labels, entry, target)
            expected = {k:full[k] for k in ('dice', *entry['terms'])}
            self.assertEqual(set(actual), set(expected))
            rows = [original.gradient_record(actual, logits)]
            selected = objectives.select_weights(rows, entry, settings)
            if entry['id'] in original.OBJECTIVES:
                self.assertEqual(selected['coefficients'], original.select_weights(rows, entry['id'], settings)['coefficients'])
            weights = selected['coefficients']
            left, right = (combined_loss(v, weights, .6) for v in (actual, expected))
            self.assertTrue(torch.equal(left, right))
            a = torch.autograd.grad(left, logits, retain_graph=True)[0]
            b = torch.autograd.grad(right, logits, retain_graph=True)[0]
            self.assertTrue(torch.equal(a, b))

    def test_winner_unchanged(self):
        labels = synthetic_target('ellipsoid')
        logits = torch.randn(1, 3, 64, 64, 64, requires_grad=True)
        name = 'synthetic'
        affine = np.eye(4)
        affine[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        item = dict(case_name=name, label=labels[0], affine=affine)
        p = dict(geometry=GEOMETRY, native_geometry={name:dict(native_shape=[64]*3)})
        entry = objectives.entries()[1]
        with tempfile.TemporaryDirectory() as directory:
            target = objectives.Targets(Path(directory), p, entry).get(item, 'cpu')
            before = combo_objectives.components(logits, labels, entry, target)
            after = objectives.components(logits, labels, entry, target)
            for name in before:
                self.assertTrue(torch.equal(before[name], after[name]))
            rows = [original.gradient_record(after, logits)]
            settings = dict(prefix_epochs=5, target_ratio=.1, cap_ratio=.5)
            self.assertEqual(objectives.select_weights(rows, entry, settings),
                             combo_objectives.select_weights(rows, entry, settings))


def gpu_preflight(p, root):
    seed_all(0)
    device = torch.device('cuda')
    train = Patients(_load_pkl_dataframe(p['pkl']), p['splits']['5pct']['train'], p['native_geometry'])
    model = model_factory(device)
    assert sum(v.numel() for v in model.parameters()) == p['model_parameters']
    item = train[0]
    labels = item['label'][None].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'],
                                 weight_decay=p['weight_decay'], foreach=False)
    checked = []
    for entry in objectives.entries():
        cache = objectives.Targets(root, p, entry)
        target = cache.get(item, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            raw = model(item['image'][None].to(device))
        z = raw.detach().float().requires_grad_(True)
        rows = [original.gradient_record(objectives.components(z, labels, entry, target), z)]
        calibration = objectives.select_weights(rows, entry, p['calibration'])
        values = objectives.components(raw, labels, entry, target)
        assert set(calibration['coefficients']) == set(values)-{'dice'}
        loss = combined_loss(values, calibration['coefficients'], 1.)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        optimizer.step()
        checked.append(entry['id'])
    atomic_json(root/'gpu_preflight.json', dict(status='passed', objectives=checked,
        finite_gradients=True, actual_optimizer_steps=len(checked), calibration='training volume only'))
    del model, optimizer, raw, z, values, loss, train, cache, target
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
