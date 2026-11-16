"""Protect no-augmentation protocol and the single-replication scope."""
from copy import deepcopy
from pathlib import Path
import unittest

from scripts.colab_hfh.medsam_native125_logged_campaign import COMPONENTS, make_protocol, pilot_phase_source
from scripts.colab_hfh.medsam_native125_logged_worker import worker_source


class NativeReplicationTests(unittest.TestCase):
    def test_original_recipe_is_unchanged(self):
        p=dict(arm='dice',fold=0,seed=0,updates=125,augmentation='none',
               frozen_sample_schedule=[['case',2]],frozen_learning_rates=[.1],train_cases=['case'])
        control=dict(protocol=p,archive_sha256='abc');saved=deepcopy(control)
        result=make_protocol(control,Path('/baseline.zip'),'none')
        for k,v in p.items():
            if k!='arm':self.assertEqual(result[k],v)
        self.assertEqual(control,saved)
        self.assertEqual(COMPONENTS,('none',))
        with self.assertRaises(ValueError):make_protocol(control,Path('/baseline.zip'),'gamma')

    def test_worker_preserves_no_augmentation_guard(self):
        source=worker_source();compile(source,'worker','exec');compile(pilot_phase_source(),'scheduler','exec')
        self.assertIn('medsam_native125_logged import install as install_aug',source)
        self.assertIn('medsam_native125_logged import install_calibration',source)
        from scripts.colab_hfh import medsam_native125_logged
        import inspect
        implementation=inspect.getsource(medsam_native125_logged.install)
        self.assertIn("protocol['augmentation'] != 'none'",implementation)
        self.assertNotIn('transform_pair(',implementation)
        self.assertNotIn('augmented_samples(',implementation)


if __name__=='__main__':unittest.main()
