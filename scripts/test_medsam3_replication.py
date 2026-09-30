import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.medsam3_replication import validate_study, paired_statistics, aggregate, required_quota_bytes


def fixture():
    supports = [[f'train_{f}_{i}' for i in range(5)] for f in range(3)]
    cases = [[f'val_{f}_{i}' for i in range(4)] for f in range(3)]
    all_cases = set(sum(supports+cases, []))
    folds = [{'train': sorted(all_cases-set(c)), 'val': c} for c in cases]
    fixed = dict(updates=200, accumulation=2, warmup_updates=20, learning_rate=5e-5,
                 final_learning_rate=5e-6, arms=['baseline', 'bands', 'bands_edge'],
                 calibration_slabs_per_volume=4, constraint_warmup_updates=20,
                 gradient_ratio_target=.1, gradient_ratio_p95_cap=.5,
                 calibration_seed=13, base_sha256='base', split_sha256='split')
    protocols = {f'fold{f}_seed{s}': dict(fixed, seed=s, fold_index=f,
                   train_cases=supports[f], evaluation_cases=cases[f]) for f in range(3) for s in (17, 83, 191)}
    study = dict(folds=[0, 1, 2], seeds=[17, 83, 191], protocols=protocols,
                 training_union=sorted(sum(supports, [])), evaluation_union=sorted(sum(cases, [])),
                 excluded_pilot_cases=['pilot'], primary_contrast=['bands_edge', 'baseline'],
                 secondary_contrasts=[['bands', 'baseline'], ['bands_edge', 'bands']],
                 bootstrap_draws=100, bootstrap_seed=13, inference_caveat='Exploratory only.')
    return study, folds


class StudyTests(unittest.TestCase):
    def test_mirrored_quota_counts_both_copies_and_keeps_reserve(self):
        study = dict(storage_replication_factor=2, reserve_gib=1.,
                     artifact_budget_bytes={'a': 3*1024**3, 'b': 2*1024**3})
        self.assertEqual(required_quota_bytes(study, ['a', 'b']), 11*1024**3)
        self.assertEqual(required_quota_bytes(study, ['b']), 5*1024**3)
        self.assertEqual(required_quota_bytes(study, []), 1024**3)
        study['storage_replication_factor'] = 1
        with self.assertRaises(ValueError):
            required_quota_bytes(study, ['a'])

    def test_valid_crossed_study(self):
        validate_study(*fixture())

    def test_global_train_evaluation_leak_is_rejected(self):
        study, folds = fixture()
        for seed in study['seeds']:
            study['protocols'][f'fold1_seed{seed}']['train_cases'][0] = 'val_0_0'
        study['training_union'] = sorted({c for p in study['protocols'].values() for c in p['train_cases']})
        with self.assertRaisesRegex(ValueError, 'another evaluation fold'):
            validate_study(study, folds)

    def test_seed_dependent_support_is_rejected(self):
        study, folds = copy.deepcopy(fixture())
        study['protocols']['fold0_seed83']['train_cases'] = ['changed']*5
        with self.assertRaisesRegex(ValueError, 'Cases vary'):
            validate_study(study, folds)

    def test_pilot_contamination_is_rejected(self):
        study, folds = fixture()
        study['excluded_pilot_cases'].append('val_0_0')
        with self.assertRaisesRegex(ValueError, 'pilot contamination'):
            validate_study(study, folds)

    def test_constant_effect_and_three_fold_p_limit(self):
        result = paired_statistics([np.full((3, n), .01) for n in (4, 7, 9)], draws=100)
        np.testing.assert_allclose(result['exploratory_hierarchical_ci95'], [.01, .01])
        self.assertAlmostEqual(result['mean_delta'], .01)
        self.assertEqual(result['fold_sign_flip_p_two_sided'], .25)
        self.assertEqual(result['positive_runs'], 9)

    def test_zero_effect_is_not_an_improvement(self):
        result = paired_statistics([np.zeros((3, 4)) for _ in range(3)], draws=100)
        self.assertEqual(result['fold_sign_flip_p_two_sided'], 1.)
        self.assertEqual(result['positive_runs'], 0)

    def test_shared_seed_effect_is_not_counted_as_nine_independent_seeds(self):
        block = np.repeat(np.array([-.02, 0., .02])[:, None], 4, axis=1)
        result = paired_statistics([block.copy() for _ in range(3)], draws=5000, seed=18)
        np.testing.assert_allclose(result['exploratory_hierarchical_ci95'], [-.02, .02])

    def test_equal_fold_weighting_and_sign_symmetry(self):
        blocks = [np.full((3, n), value) for n, value in ((4, .01), (7, -.02), (9, .04))]
        forward = paired_statistics(blocks, draws=1000, seed=42)
        reverse = paired_statistics([-b for b in blocks], draws=1000, seed=42)
        self.assertAlmostEqual(forward['mean_delta'], .01)
        np.testing.assert_allclose(forward['exploratory_hierarchical_ci95'],
                                   -np.asarray(reverse['exploratory_hierarchical_ci95'])[::-1])

    def test_aggregation_requires_complete_audited_results(self):
        study, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study_path = root/'study.json'
            study_path.write_text(json.dumps(study))
            with self.assertRaises(FileNotFoundError):
                aggregate(study_path, root/'results', root/'analysis')
            for name, protocol in study['protocols'].items():
                path = root/'results'/name
                path.mkdir(parents=True)
                means = dict(baseline=.8, bands=.81, bands_edge=.82)
                record = dict(status='complete', protocol=protocol, mean_dice=means,
                              per_case={c: {a: {'dice': v} for a, v in means.items()} for c in protocol['evaluation_cases']},
                              calibration={a: {'weight': 1.} for a in ('bands', 'edge')})
                raw = json.dumps(record).encode()
                (path/'results.json').write_bytes(raw)
                (path/'audit.json').write_text(json.dumps(dict(status='passed', results_sha256=hashlib.sha256(raw).hexdigest())))
            report = aggregate(study_path, root/'results', root/'analysis')
            self.assertEqual(len(report['runs']), 9)
            self.assertAlmostEqual(report['contrasts']['bands_edge_minus_baseline']['mean_delta'], .02)
            path = root/'results/fold0_seed17/results.json'
            record = json.loads(path.read_text())
            record['calibration']['edge']['weight'] = 1.001
            path.write_text(json.dumps(record))
            audit_path = path.parent/'audit.json'
            audit_path.write_text(json.dumps(dict(status='passed', results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            with self.assertRaisesRegex(ValueError, 'strict study gate failed'):
                aggregate(study_path, root/'results', root/'strict')
            observed = aggregate(study_path, root/'results', root/'observed', report_calibration_deviation=True)
            self.assertEqual(observed['status'], 'reported_with_calibration_deviation')
            self.assertFalse(observed['strict_study_gate_passed'])
            self.assertEqual(len(observed['runs']), 9)
            self.assertEqual(observed['contrasts'], report['contrasts'])
            self.assertIn('strict cross-seed calibration gate did not pass', (root/'observed/RESULTS.md').read_text())
            shared = copy.deepcopy(study)
            for seed in shared['seeds']:
                name = f'fold0_seed{seed}'
                shared['protocols'][name]['calibration_source_run'] = 'fold0_seed17'
                rp = root/'results'/name/'results.json'
                rr = json.loads(rp.read_text()); rr['protocol'] = shared['protocols'][name]
                rp.write_text(json.dumps(rr))
                (rp.parent/'audit.json').write_text(json.dumps(dict(status='passed', results_sha256=hashlib.sha256(rp.read_bytes()).hexdigest())))
            study_path.write_text(json.dumps(shared))
            with self.assertRaisesRegex(ValueError, 'not exactly identical'):
                aggregate(study_path, root/'results', root/'observed', report_calibration_deviation=True)
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                aggregate(study_path, root/'results', root/'analysis')
            with self.assertRaisesRegex(ValueError, 'Unverified'):
                aggregate(study_path, root/'results', root/'observed', report_calibration_deviation=True)


if __name__ == '__main__':
    unittest.main()
