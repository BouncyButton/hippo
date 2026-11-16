"""Frozen split/schedule and non-leaking calibration tests for replication."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.colab_hfh.medsam_gain_replication import sample_prefix,make_protocol,check_protocol,install_calibration,ARMS,PAIRS
from scripts.colab_hfh.medsam_gain_replication_worker import worker_source
from scripts.colab_hfh.medsam_gain_replication_campaign import phase_source

class GainReplicationTests(unittest.TestCase):
    def reference(self):
        return dict(updates=200,seed=1,fold=2,name='f2s1',accumulation=2,
            train_cases=[f'train{i}_h{h}' for i in range(5) for h in (1,2)],
            evaluation_cases=[f'val{i}_h{h}' for i in range(5) for h in (1,2)],
            support_subjects=[f'train{i}' for i in range(5)],
            base_sha256='base',lora_sha256='lora',dataset_sha256='data',learning_rate=5e-5,
            final_learning_rate=5e-6,warmup_updates=20,calibration_seed=20260929)

    def test_native_and_constraints_share_samples_and_disjoint_subjects(self):
        ref=self.reference();schedule=sample_prefix(ref,{c:32 for c in ref['train_cases']})
        self.assertEqual(len(schedule),250)
        for arm in ARMS:
            pilot=dict(ref,updates=125,fold=0,seed=0,arm=arm,constraint_recipe={'name':arm},
                frozen_learning_rates=[.1]*200,augmentation_control_archive='/old',augmentation_control_sha256='old')
            snapshot=deepcopy(pilot)
            p=make_protocol(ref,pilot,schedule,arm);check_protocol(p)
            self.assertEqual(p['frozen_sample_schedule'],schedule)
            self.assertEqual(p['constraint_recipe'],pilot['constraint_recipe'])
            self.assertEqual(p['frozen_learning_rates'],pilot['frozen_learning_rates'])
            self.assertNotIn('augmentation_control_archive',p)
            self.assertEqual(pilot,snapshot)
            bad=deepcopy(ref);bad['evaluation_cases'][0]=bad['train_cases'][0]
            with self.assertRaises(ValueError):make_protocol(bad,pilot,schedule,arm)
        self.assertEqual(len(PAIRS),9)

    def test_seed_changes_tape_and_all_jobs_are_allowed(self):
        ref=self.reference();counts={c:32 for c in ref['train_cases']}
        self.assertNotEqual(sample_prefix(ref,counts),sample_prefix(dict(ref,seed=0),counts))
        compile(worker_source(),'worker','exec');compile(phase_source(),'scheduler','exec')
        self.assertIn('medsam_gain_replication import install_calibration',worker_source())
        self.assertIn('self.existing(job)',phase_source())

    def test_fresh_calibration_does_not_open_fold_zero_archive(self):
        engine=SimpleNamespace(write=lambda *a:None,calibrate=lambda *a:{'weights':{}})
        p={'calibration_policy':'fresh','fold':1,'seed':1}
        with patch('torch.cuda.get_device_name',return_value='test GPU'),patch('torch.cuda.is_bf16_supported',return_value=True), \
             patch('scripts.colab_hfh.medsam_intensity_augmentation.install_calibration',side_effect=AssertionError('Must not reuse fold0 calibration')):
            install_calibration(engine,p,Path('/tmp'))
        self.assertTrue(callable(engine.calibrate))

if __name__=='__main__':unittest.main()
