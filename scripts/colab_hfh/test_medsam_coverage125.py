"""Sampling scope and pairing invariants without loading a foundation model."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np

from scripts.colab_hfh.medsam_coverage125 import schedules, make_protocol
from scripts.colab_hfh.medsam_coverage125_worker import worker_source


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.flags={f'case_{i}':[5<=z<20+i%4 for z in range(32)] for i in range(10)}

    def test_coverage_and_budget(self):
        train,cal,audit=schedules(self.flags)
        self.assertEqual(len(train),250)
        self.assertEqual(len(set(map(tuple,train))),250)
        self.assertEqual(len(cal),40)
        for c,flags in self.flags.items():
            seen={z for case,z in train if case==c}
            self.assertEqual(len(seen),25)
            self.assertTrue({z for z,v in enumerate(flags) if v}.issubset(seen))
            self.assertTrue(any(not flags[z] for z in seen))
            self.assertEqual(sum(case==c for case,z in cal),4)
        self.assertTrue(set(map(tuple,cal)).issubset(set(map(tuple,train))))
        self.assertEqual((train,cal,audit),schedules(self.flags))
        self.assertNotEqual(train,schedules(self.flags,1)[0])

    def test_pair_keeps_recipe_and_initialization(self):
        p=dict(fold=0,seed=0,updates=125,accumulation=2,augmentation='none',arm='dice',
               frozen_learning_rates=[.001]*200,learning_rate=.001,train_cases=list(self.flags),
               evaluation_cases=['held_out'],base_sha256='base',lora_sha256='lora')
        before=deepcopy(p)
        a=make_protocol(p,self.flags,'dice',{'gpu':'test'})
        b=make_protocol(p,self.flags,'dice_edge_depth',{'gpu':'test'})
        self.assertEqual(p,before)
        for key in ('frozen_sample_schedule','frozen_calibration_schedule','frozen_learning_rates',
                    'base_sha256','lora_sha256','evaluation_cases','train_cases','paired_execution','augmentation','updates'):
            self.assertEqual(a[key],b[key])
        self.assertTrue(a['constraint_recipe']['native_control'])
        with self.assertRaises(ValueError):make_protocol({**p,'seed':1},self.flags,'dice',{})
        with self.assertRaises(ValueError):make_protocol(p,self.flags,'dice_bands',{})

    def test_worker_hooks(self):
        source=worker_source()
        ast.parse(source)
        self.assertIn('SupportSlabs = coverage_slabs()',source)
        self.assertIn('from scripts.colab_hfh.medsam_coverage125 import install_calibration',source)
        self.assertIn('from scripts.colab_hfh.medsam_native125_logged import install as install_aug',source)
        self.assertNotIn('from scripts.colab_hfh.medsam_native125_logged import install_calibration',source)


if __name__=='__main__':unittest.main()
