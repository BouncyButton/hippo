"""Decision, recovery, calibration and source-composition tests without a GPU."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from scripts.colab_hfh import medsam_adaptive_band as adaptive


def observation(fraction=.6, subjects=4):
    return dict(total=dict(fraction=fraction), subjects_ready=subjects)


class AdaptiveTests(unittest.TestCase):
    def test_two_consecutive_checks_required(self):
        s,e = adaptive.transition(adaptive.empty_state(),observation(),0)
        self.assertIsNone(e); self.assertFalse(s['active'])
        s,e = adaptive.transition(s,observation(),20)
        self.assertEqual(e,'activate'); self.assertEqual(adaptive.band_factor(s,20),.05)
        self.assertEqual(adaptive.band_factor(s,39),1.)

    def test_no_forced_activation(self):
        s = adaptive.empty_state()
        for u in range(0,125,20):
            s,e = adaptive.transition(s,observation(.49,5),u)
            self.assertFalse(s['active']); self.assertEqual(adaptive.band_factor(s,u),0)

    def test_one_subject_cannot_dominate(self):
        s = adaptive.empty_state()
        for u in (0,20,40):
            s,e = adaptive.transition(s,observation(.99,2),u)
        self.assertFalse(s['active'])

    def test_interrupted_streak(self):
        s = adaptive.empty_state()
        for u,f in ((0,.6),(20,.3),(40,.6)):
            s,e = adaptive.transition(s,observation(f),u)
        self.assertFalse(s['active']); self.assertEqual(s['streak'],1)

    def test_hysteresis_and_reactivation(self):
        s = adaptive.empty_state()
        for u in (0,20):
            s,_ = adaptive.transition(s,observation(),u)
        s,_ = adaptive.transition(s,observation(.5),40)
        self.assertTrue(s['active'])
        s,e = adaptive.transition(s,observation(.44),60)
        self.assertEqual(e,'deactivate'); self.assertEqual(s['weight'],0)
        s,_ = adaptive.transition(s,observation(),80)
        s,e = adaptive.transition(s,observation(),100)
        self.assertEqual(e,'activate'); self.assertEqual(adaptive.band_factor(s,100),.05)

    def test_no_errors_do_not_trigger(self):
        s = adaptive.empty_state()
        for u in (0,20):
            s,_ = adaptive.transition(s,observation(None),u)
        self.assertFalse(s['active'])

    def test_checkpoint_round_trip(self):
        import json
        s,_ = adaptive.transition(adaptive.empty_state(),observation(),0)
        restored = json.loads(json.dumps(s))
        self.assertEqual(adaptive.transition(s,observation(),20),adaptive.transition(restored,observation(),20))

    def test_transition_does_not_mutate_checkpoint(self):
        s = adaptive.empty_state(); before = deepcopy(s)
        adaptive.transition(s,observation(),0)
        self.assertEqual(s,before)

    def test_remote_errors_in_denominator(self):
        rows = [dict(case=f's{i}_h{j}',fp=100,fn=20,outer_fp=10,inner_fn=10,empty_slice_fp=70)
                for i in range(5) for j in range(2)]
        result = adaptive.summarize_errors(rows)
        self.assertAlmostEqual(result['total']['fraction'],1/6)
        self.assertEqual(result['total']['empty_slice_fp'],700)
        self.assertEqual(result['subjects_ready'],0)

    def test_missing_crops_fail(self):
        with self.assertRaises(ValueError):
            adaptive.summarize_errors([])

    def test_calibration_zero_rows_retained(self):
        r = [0.]*60+[1.]*20
        self.assertAlmostEqual(adaptive.coefficient(r,'bands'),.05)
        self.assertAlmostEqual(adaptive.coefficient([1.]*80,'bands'),.01)

    def test_p95_caps_high_tail(self):
        r = [1.]*70+[100.]*10
        w = adaptive.coefficient(r,'bands')
        self.assertLessEqual(np.quantile(r,.95)*w,.05+1e-12)

    def test_no_gradient_support_fails(self):
        with self.assertRaises(ValueError):
            adaptive.coefficient([0.]*80,'bands')

    def test_schedule_training_only_and_reproducible(self):
        p = dict(train_cases=[f's{i}_h{j}' for i in range(5) for j in range(2)],
                 evaluation_cases=['heldout'],calibration_seed=123,fold=1)
        counts = dict.fromkeys(p['train_cases'],32)
        a = adaptive.calibration_schedule(p,counts)
        self.assertEqual(a,adaptive.calibration_schedule(p,counts)); self.assertEqual(len(a),80)
        self.assertEqual({r['case'] for r in a},set(p['train_cases']))
        with self.assertRaises(ValueError):
            adaptive.calibration_schedule(p,{**counts,'heldout':32})

    def test_subject_bootstrap_reproducible(self):
        rows = [dict(subject=f's{i}',native_norm2=1.,auxiliary_norm2=float(j%3))
                for i in range(5) for j in range(16)]
        a = adaptive.calibration_summary(rows,'bands')
        self.assertEqual(a,adaptive.calibration_summary(rows,'bands'))
        self.assertEqual(a['zeros'],30)

    def test_composed_training_and_worker_hooks(self):
        from scripts.cross_dataset_campaign import foundation
        from scripts.colab_hfh.medsam_intensity_augmentation import training_source as augmented
        from scripts.colab_hfh.medsam_adaptive_band_worker import worker_source
        from scripts.colab_hfh.medsam_adaptive_band_campaign import validator
        source = adaptive.training_source(inspect.getsource(foundation.train))
        compile(augmented(source),'test_composed','exec')
        self.assertIn("hfh_gate.restore(state['adaptive_band'])",source)
        self.assertIn('adaptive_band=deepcopy(hfh_gate.state)',source)
        worker = worker_source(); compile(worker,'worker','exec'); validator()
        self.assertIn('medsam_adaptive_band_recovery',worker)

    def test_install_source_inspection_survives(self):
        from pathlib import Path
        from types import SimpleNamespace
        from scripts.cross_dataset_campaign import foundation
        from scripts.colab_hfh import medsam_gain_replication
        engine = SimpleNamespace(train=foundation.train,write=lambda *a:None)
        def gain_install(e,p,o):
            source = inspect.getsource(e.train)
            self.assertIn('hfh_gate.before_update',source)
        with patch.object(adaptive,'check_protocol'), patch.object(medsam_gain_replication,'install',gain_install):
            adaptive.install(engine,{},Path('/unused'))


if __name__ == '__main__':
    unittest.main()
