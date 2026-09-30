import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.medsam3_calibration_cache import calibration_context, load_calibration
from scripts.medsam3_replication import validate_study


class CalibrationCacheTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(support='fixed', adapter_before='adapter', frozen_before='base')
        self.report = dict(context=self.context, source_run='fold3_seed17', evaluation_labels_used=False,
                           bands={'weight': 13.}, edge={'weight': 300.})

    def load(self, report, context=None, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'calibration.json'
            p.write_text(json.dumps(report))
            return load_calibration(p, expected or hashlib.sha256(p.read_bytes()).hexdigest(),
                                    context or self.context, 'fold3_seed17')

    def test_reuses_identical_report(self):
        self.assertEqual(self.load(self.report), self.report)

    def test_rejects_changed_file_context_or_validation_use(self):
        with self.assertRaisesRegex(ValueError, 'file changed'):
            self.load(self.report, expected='wrong')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.load(self.report, context=dict(support='different'))
        report = copy.deepcopy(self.report)
        report['evaluation_labels_used'] = True
        with self.assertRaisesRegex(ValueError, 'training-only'):
            self.load(report)

    def test_rejects_invalid_coefficient(self):
        for value in (0, -1, float('nan'), float('inf')):
            report = copy.deepcopy(self.report)
            report['bands']['weight'] = value
            with self.assertRaisesRegex(ValueError, 'coefficient'):
                self.load(report)

    def test_context_ignores_training_seed_but_binds_support_and_settings(self):
        protocol = dict(train_cases=['a'], base_sha256='base', calibration_seed=5,
                        calibration_slabs_per_volume=4, gradient_ratio_target=.1,
                        gradient_ratio_p95_cap=.5, seed=17)
        first = calibration_context(protocol, [{'image_sha256': 'image'}], 'adapter', 'frozen')
        protocol['seed'] = 83
        self.assertEqual(first, calibration_context(protocol, [{'image_sha256': 'image'}], 'adapter', 'frozen'))
        self.assertNotEqual(first, calibration_context(protocol, [{'image_sha256': 'changed'}], 'adapter', 'frozen'))
        protocol['train_cases'] = ['b']
        self.assertNotEqual(first, calibration_context(protocol, [{'image_sha256': 'image'}], 'adapter', 'frozen'))

    def test_frozen_extension_keeps_prior_runs_and_all_roles_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        prior = json.loads((root/'docs/experiments/medsam3_replication_20260928/protocols/study.json').read_text())
        study = json.loads((root/'docs/experiments/medsam3_fold3_extension_20260929/protocols/study.json').read_text())
        folds = json.loads((root/'datasets/Dataset101_MSD/splits_final.json').read_text())
        validate_study(study, folds)
        for name, protocol in prior['protocols'].items():
            self.assertEqual(study['protocols'][name], protocol)
        new = study['protocols']['fold3_seed17']
        self.assertEqual(len(new['train_cases']), 5)
        self.assertEqual(len(new['evaluation_cases']), 44)
        self.assertFalse(set(new['train_cases']) & set(prior['training_union']+prior['evaluation_union']))
        self.assertEqual(new['calibration_source_run'], 'fold3_seed17')
        self.assertEqual(study['seeds'], [17, 83, 191])
        self.assertEqual(study['folds'], [0, 1, 2, 3])
        study['protocols']['fold3_seed83']['calibration_source_run'] = 'fold0_seed17'
        with self.assertRaisesRegex(ValueError, 'policy varies'):
            validate_study(study, folds)


if __name__ == '__main__':
    unittest.main()
