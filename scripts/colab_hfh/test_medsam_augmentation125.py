"""Check pairing, resume determinism, and label/evaluation isolation on CPU."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.colab_hfh.medsam_intensity_augmentation import (
    SPEC, augmentation_seed, augmented_samples, transform_pair, training_source, check_protocol)
from scripts.colab_hfh.medsam_augmentation125_campaign import make_protocol, pilot_phase_source
from scripts.colab_hfh.medsam_augmentation125_worker import worker_source


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(4096, dtype=np.uint16).reshape(64,64).astype(np.uint8)
        self.label = (self.image > 180).astype(np.uint8)

    def test_resume_and_arm_pairing_do_not_use_global_rng(self):
        protocol = dict(seed=0, fold=0)
        seed = augmentation_seed(protocol, 73, 1, 'case', 14)
        np.random.seed(5)
        before = np.random.get_state()
        a, _ = transform_pair([self.image]*2, seed)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])
        np.random.seed(991)
        b, _ = transform_pair([self.image]*2, seed)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)
        self.assertNotEqual(seed, augmentation_seed(protocol,74,1,'case',14))

    def test_samples_and_labels_restore_even_after_exception(self):
        saved = [(self.image.copy(),self.label.copy()) for _ in range(3)]
        slabs = SimpleNamespace(indices={'case':[0,1,2]}, samples=list(saved))
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with augmented_samples(slabs,'case',2,1234):
                self.assertIs(slabs.samples[2][1],saved[2][1])
                self.assertIs(slabs.samples[0][1],saved[0][1])
                self.assertIs(slabs.samples[1],saved[1])
                raise RuntimeError('test')
        for a,b in zip(slabs.samples,saved):
            self.assertIs(a,b)

    def test_physical_grid_and_zero_padding_unchanged(self):
        changed = 0
        for seed in range(20):
            images, params = transform_pair([self.image]*2, seed)
            for image in images:
                self.assertEqual(image.shape,self.image.shape)
                self.assertEqual(image.dtype,np.uint8)
                self.assertTrue(np.all(image[self.image==0]==0))
                changed += int(not np.array_equal(image,self.image))
        self.assertGreater(changed,10)

    def test_protocol_changes_only_declared_augmentation_fields(self):
        p=dict(arm='dice_bands',fold=0,seed=0,updates=125,augmentation='none',
               frozen_sample_schedule=[['case',2]],frozen_learning_rates=[.1],
               constraint_recipe={'name':'unchanged'},train_cases=['case'])
        control=dict(protocol=p,archive_sha256='abc')
        snapshot=deepcopy(control)
        changed=make_protocol(control,Path('/control.zip'))
        self.assertEqual(control,snapshot)
        for key in p.keys()-{'arm','augmentation'}:
            self.assertEqual(p[key],changed[key])
        check_protocol(changed)
        changed['updates']=200
        with self.assertRaises(ValueError):
            check_protocol(changed)

    def test_actual_training_loop_hook_and_worker_compile(self):
        from scripts.cross_dataset_campaign import foundation
        source=training_source(inspect.getsource(foundation.train))
        self.assertIn('hfh_augmented_forward(model,criterion,slabs,case,anchor,update,micro)',source)
        compile(source,'train','exec')
        compile(worker_source(),'worker','exec')
        compile(pilot_phase_source(),'scheduler','exec')
        self.assertNotIn('200',pilot_phase_source())


if __name__ == '__main__':
    unittest.main()
