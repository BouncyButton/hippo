"""Recovery regressions: reject drift; restore the complete common training state."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.depth_supervision.utils import seed_all
from scripts.edge_ablation import train, recover
from scripts.edge_ablation.objectives import experiment_matrix
from scripts.edge_ablation.tests import TinyPatients, tiny_model


class RecoveryTests(unittest.TestCase):
    def test_original_numeric_guard_is_not_relaxed(self):
        row = dict(epoch=1, constraint_scale=0, train_bands=0, train_edge=0,
                   coefficient_bands=0, coefficient_edge=0, train_dice=.8, validation_dice=.14)
        self.assertEqual(recover.check_prefix([row], [row], 1)[0]['validation_dice'], 0)
        with self.assertRaisesRegex(ValueError, 'Unmatched Dice warmup'):
            recover.check_prefix([dict(row, validation_dice=.14-.00034124872)], [row], 1)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            recover.check_prefix([], [row], 1)
        with self.assertRaisesRegex(ValueError, 'Constraint active'):
            recover.check_prefix([dict(row, constraint_scale=.2)], [row], 1)
        with self.assertRaisesRegex(ValueError, 'Unmatched band coefficient'):
            recover.check_band(.002145659159035607, .0021464975081461886)

    def test_shared_prefix_recovers_changed_initialization_and_rng(self):
        p = dict(identity_sha='recovery-test', splits={'5pct': dict(train=['a', 'b'], val=['v'])}, pkl='unused',
                 native_geometry={n: dict(native_shape=[16]*3) for n in ('a', 'b', 'v')}, geometry={},
                 learning_rate=.001, weight_decay=1e-5, step_size=3, lr_gamma=.5,
                 calibration=dict(prefix_epochs=2, ramp_epochs=2, target_ratio=.1, cap_ratio=.5, max_cases=10),
                 max_epochs=5, early_stopping=dict(patience=0, min_epochs=1, min_delta=0.))
        base = dict(experiment_matrix()[0], fraction='5pct', seed=0, arm='ordinary_swinunetr')
        entry = dict(experiment_matrix()[-1], fraction='5pct', seed=0, arm='ordinary_swinunetr')
        p['runs'] = [base, entry]
        with tempfile.TemporaryDirectory() as directory, patch.object(train, '_load_pkl_dataframe', return_value=None), \
                patch.object(train, 'Patients', TinyPatients), patch.object(train, 'model_factory', side_effect=tiny_model), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            repair = root/'repair'
            repair.mkdir()
            train.run(p, root, base, device_override='cpu')
            expected = train.run(p, root/'reference', entry, device_override='cpu')
            value = recover.prefix_probe(p, root, repair, 1, device='cpu')
            # A deliberately different initialization/RNG reproduces the mismatch.
            # Restoring the shared checkpoint must make it irrelevant.
            with patch.object(train, 'seed_all', side_effect=lambda seed: seed_all(seed+1)):
                train.run(p, root/'bad', entry, stop_after=2, device_override='cpu')
                with self.assertRaisesRegex(ValueError, 'Unmatched Dice warmup'):
                    recover.check_prefix(recover.history(root/'bad/runs'/entry['id']),
                                         recover.history(root/'runs/bands'), 2)
                recover.seed_run(p, root, entry, value)
                actual = train.run(p, root, entry, device_override='cpu')
            self.assertEqual(expected['validation'], actual['validation'])
            recover.check_completed(p, root, entry)
            a = torch.load(root/'reference/runs'/entry['id']/'best.pt', weights_only=False)
            b = torch.load(root/'runs'/entry['id']/'best.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
            for left, right in zip(recover.history(root/'reference/runs'/entry['id']), recover.history(root/'runs'/entry['id'])):
                remove = ('epoch_seconds', 'elapsed_seconds')
                self.assertEqual({k:v for k,v in left.items() if k not in remove},
                                 {k:v for k,v in right.items() if k not in remove})
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                recover.seed_run(p, root, entry, value)
            bad = copy.deepcopy(value)
            bad['artifacts']['latest.pt'] = 'tampered'
            with self.assertRaisesRegex(ValueError, 'Verified prefix changed'):
                recover.seed_run(p, root/'other', entry, bad)
            from scripts.edge_ablation.report import report
            self.assertEqual(report(p, root)['completed'], 2)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
