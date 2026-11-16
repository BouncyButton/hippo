"""Pairing and resource admission for the four authorized constraint runs."""
from copy import deepcopy
from pathlib import Path
import unittest
from scripts.colab_hfh.medsam_component_constraints_campaign import (
    ARMS, PAIRS, ThreeWorkers, make_protocol, pilot_phase_source)
from scripts.colab_hfh.medsam_augmentation_components import component_spec

class PairingTests(unittest.TestCase):
    def controls(self, component, arm):
        p=dict(arm=arm,fold=0,seed=0,updates=125,augmentation='none',
            frozen_sample_schedule=[['a',1]],frozen_learning_rates=[.1]*200,
            train_cases=['a'],evaluation_cases=['b'],dataset_sha256='data',
            base_sha256='base',lora_sha256='lora',constraint_recipe={'name':'fixed'},
            gradient_ratio_targets={'edge':.1},constraint_warmup_updates=20)
        q=deepcopy(p);q.update(arm='dice',augmentation=component_spec(component))
        return dict(protocol=p,archive_sha256='constraint'),dict(protocol=q,archive_sha256='native')

    def test_all_four_reuse_distinct_native_and_calibration_sources(self):
        self.assertEqual(len(PAIRS),4)
        for component,arm in PAIRS:
            reference,control=self.controls(component,arm)
            before=deepcopy((reference,control))
            p=make_protocol(reference,Path('/constraint.zip'),control,Path('/native.zip'),component,arm)
            self.assertEqual(p['augmentation_control_sha256'],'constraint')
            self.assertEqual(p['baseline_archive_sha256'],'native')
            self.assertEqual(p['augmentation'],control['protocol']['augmentation'])
            for k in ('constraint_recipe','gradient_ratio_targets','constraint_warmup_updates','updates','frozen_learning_rates','frozen_sample_schedule'):
                self.assertEqual(p[k],reference['protocol'][k])
            self.assertEqual((reference,control),before)
            control['protocol']['frozen_sample_schedule']=[['different',1]]
            with self.assertRaises(ValueError):make_protocol(reference,Path('/r'),control,Path('/c'),component,arm)

    def test_wrong_component_rejected(self):
        reference,control=self.controls('gain',ARMS[0])
        with self.assertRaises(ValueError):make_protocol(reference,Path('/r'),control,Path('/c'),'noise',ARMS[0])

    def test_three_workers_keep_memory_and_loading_guards(self):
        tuner=ThreeWorkers()
        gpu=dict(total_gib=40,used_gib=20,host_free_gib=60,cpu_slots=3,processes={1:10,2:10},utilization=90)
        workers=[dict(pid=i,name=str(i),step=10,phase='training',peak_gib=10) for i in (1,2)]
        tuner.observe(1,gpu,workers)
        self.assertEqual(tuner.target,3);self.assertTrue(tuner.admission(gpu,workers))
        self.assertFalse(tuner.admission(dict(gpu,used_gib=35),workers))
        self.assertFalse(tuner.admission(gpu,[dict(workers[0],step=0),workers[1]]))
        l4=dict(gpu,total_gib=22.5,used_gib=0,cpu_slots=3)
        tuner.observe(2,l4,[])
        self.assertLess(tuner.target,3)
        source=pilot_phase_source();compile(source,'scheduler','exec')
        self.assertIn('self.tuner=ThreeWorkers()',source)
        self.assertIn('self.prepare(component,arm,0,0)',source)
        self.assertIn('self.existing(job)',source)
        self.assertIn('self.commit(job)',source)

if __name__=='__main__':unittest.main()
