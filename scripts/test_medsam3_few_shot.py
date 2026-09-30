"""Guardrails for support selection, targets, and fixed-budget few-shot scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from scripts import train_medsam3_few_shot as runner


class FewShotTests(unittest.TestCase):
    def test_protocol_rejects_leakage_and_wrong_splits(self):
        p = dict(train_cases=list('abcde'), evaluation_cases=['f'], updates=2, accumulation=1)
        split = dict(train=list('abcde'), val=['f'])
        runner.validate_protocol(p, split)
        for update in [dict(evaluation_cases=['a']), dict(evaluation_cases=['g']),
                       dict(train_cases=list('aabcd')), dict(updates=0)]:
            with self.assertRaises(ValueError):
                runner.validate_protocol(p | update, split)

    def test_box_uses_xy_exclusive_edges_and_handles_empty(self):
        mask = np.zeros((10, 20), bool)
        self.assertIsNone(runner.target_box(mask))
        mask[2:6, 3:10] = True
        np.testing.assert_allclose(runner.target_box(mask), [6.5/20, 4/10, 7/20, 4/10])
        mask[:] = True
        np.testing.assert_allclose(runner.target_box(mask), [.5, .5, 1, 1])

    def test_schedule_has_fixed_warmup_and_final_lr(self):
        lr = [runner.lr_for_update(i, 200, 20, 5e-5, 5e-6) for i in range(200)]
        self.assertAlmostEqual(lr[0], 2.5e-6)
        self.assertAlmostEqual(lr[19], 5e-5)
        self.assertAlmostEqual(lr[-1], 5e-6)
        self.assertTrue(all(a >= b for a, b in zip(lr[20:], lr[21:])))

    def test_scoring_cannot_open_any_reference_until_all_cases_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4))
            names = ('zero_shot', 'few_shot', 'few_shot_bands', 'few_shot_bands_edge')
            runner.save_predictions(root, 'case_a', source, {n:np.zeros(source.shape) for n in names})
            with mock.patch.object(runner, 'load_label') as label:
                with self.assertRaises(RuntimeError):
                    runner.score_saved_predictions(root, ['case_a', 'case_b'], root)
                label.assert_not_called()
            with self.assertRaises(FileExistsError):
                runner.save_predictions(root, 'case_a', source, {'few_shot':np.zeros(source.shape)})

    def test_real_selection_is_frozen_and_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root/'docs/experiments/medsam3_supervised_5shot_20260928/protocol.json').read_text())
        path = root/'datasets/Dataset101_MSD/splits_final.json'
        if not path.exists():
            self.skipTest('Dataset split is not present in packaged source.')
        self.assertEqual(runner.sha256(path), protocol['split_sha256'])
        runner.validate_protocol(protocol, json.loads(path.read_text())[0])


if __name__ == '__main__':
    unittest.main()
