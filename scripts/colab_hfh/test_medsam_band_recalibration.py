"""CPU checks for calibration distribution, tail bounds and scientific pairing."""
from copy import deepcopy
from pathlib import Path
import random
import unittest

import numpy as np

from scripts.colab_hfh.medsam_band_recalibration import POLICY, calibration_schedule, coefficient, summarize
from scripts.colab_hfh.medsam_band_recalibration_worker import worker_source
from scripts.colab_hfh.medsam_band_recalibration_campaign import validator


class CalibrationTests(unittest.TestCase):
    def protocol(self):
        return dict(train_cases=[f'subject_{s}_h{h}' for s in range(5) for h in (1,2)],
                    evaluation_cases=['held_out_h1'], calibration_seed=20260929, fold=0, seed=0)

    def test_sampler_covers_training_distribution_without_rng_side_effects(self):
        p = self.protocol()
        state = random.getstate()
        npstate = np.random.get_state()
        schedule = calibration_schedule(p, {c:32 for c in p['train_cases']})
        self.assertEqual(len(schedule), 200)
        self.assertEqual(schedule, calibration_schedule(p, {c:32 for c in p['train_cases']}))
        self.assertEqual(random.getstate(), state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], npstate[1]))
        for c in p['train_cases']:
            rows = [r for r in schedule if r['case'] == c]
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(0 <= r['anchor'] < 32 for r in rows))
            self.assertLess(min(r['anchor'] for r in rows), 2)
            self.assertGreater(max(r['anchor'] for r in rows), 29)
        with self.assertRaises(ValueError):
            calibration_schedule(p, {**{c:32 for c in p['train_cases']}, 'held_out_h1':32})

    def test_tail_cap_binds_for_bimodal_distribution(self):
        ratios = np.array([.001]*100 + [.1]*100)
        w = coefficient(ratios)
        self.assertLessEqual(np.median(ratios)*w, .01 + 1e-12)
        self.assertLessEqual(np.quantile(ratios, .95)*w, .05 + 1e-12)
        heavy_tail = np.array([.001]*180 + [.1]*20)
        self.assertAlmostEqual(coefficient(heavy_tail), .5)

    def test_empty_slabs_are_retained_and_zero_median_supported(self):
        ratios = np.array([0.]*150 + [.1]*50)
        self.assertAlmostEqual(coefficient(ratios), .5)
        with self.assertRaises(ValueError):
            coefficient([0.]*200)
        for bad in ([np.nan, .1], [-.1, .1], [np.inf, .1]):
            with self.assertRaises(ValueError):
                coefficient(bad)

    def test_stability_is_diagnostic_and_subject_clustered(self):
        p = self.protocol()
        rows = calibration_schedule(p, {c:32 for c in p['train_cases']})
        ratios = []
        for i, row in enumerate(rows):
            ratio = 0. if i%3 == 0 else .002*(1+i//40)
            ratios.append(ratio)
            row.update(gram=[[100.,0.],[0.,100.*ratio**2]], slab_category='empty' if ratio==0 else 'foreground')
        result = summarize(rows)
        self.assertAlmostEqual(result['weights']['bands'], coefficient(ratios))
        self.assertEqual(result['statistics']['bands']['zero_band_gradients'], 67)
        self.assertEqual(len(result['stability']['leave_one_subject_out']), 5)
        self.assertFalse(result['stability']['used_to_select_weight'])
        self.assertEqual(result, summarize(deepcopy(rows)))

    def test_worker_and_validator_hooks_compile(self):
        source = worker_source()
        compile(source, '<recalibration-worker>', 'exec')
        self.assertEqual(source.count('from scripts.colab_hfh.medsam_band_recalibration import'), 2)
        self.assertIn('original.main()', source)
        self.assertTrue(callable(validator()))


if __name__ == '__main__':
    unittest.main()
