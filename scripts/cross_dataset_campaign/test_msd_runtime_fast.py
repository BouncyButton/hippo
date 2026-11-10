"""CPU invariants for the runtime-only MSD acceleration hooks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import read
from scripts.cross_dataset_campaign.msd_mixed_losses_job import ROOT,NAME
from scripts.cross_dataset_campaign.msd_runtime_fast import RuntimeAcceleration,evaluate


class RuntimeInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.protocol=read(ROOT/'experiments'/NAME/'protocols/swinunetr/protocol.json')
        x,y,z=torch.meshgrid(*(torch.arange(64),)*3,indexing='ij')
        fg=((x-32)/18)**2+((y-32)/12)**2+((z-32)/10)**2<1
        cls.labels=torch.where(fg,torch.where(x<32,1,2),0)[None,None]

    def test_metric_and_rng_equivalence(self):
        torch.manual_seed(8)
        cases=[dict(image=torch.randn(1,64,64,64),label=self.labels[0],case_name=str(i)) for i in range(3)]
        model=torch.nn.Conv3d(1,3,1)
        before=torch.get_rng_state()
        expected=experiment.evaluate(model,cases,torch.device('cpu'),False,self.protocol['classes'])
        after=torch.get_rng_state()
        torch.set_rng_state(before)
        actual=evaluate(model,cases,torch.device('cpu'),False,self.protocol['classes'])
        self.assertEqual(expected,actual)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_annotation_cache_and_mutation_invalidation(self):
        label=self.labels.clone()
        affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
        sdf=experiment.trainer.signed_distance_maps
        depth=experiment.trainer.geometry_weights
        expected_sdf=sdf(label);expected_depth=depth(label,affine,self.protocol['depth_geometry'])
        with RuntimeAcceleration(experiment,self.protocol,'cpu') as fast:
            actual_sdf=experiment.trainer.signed_distance_maps(label)
            actual_depth=experiment.trainer.geometry_weights(label,affine,self.protocol['depth_geometry'])
            self.assertTrue(torch.equal(expected_sdf,actual_sdf))
            self.assertTrue(torch.equal(expected_depth,actual_depth))
            self.assertIs(actual_sdf,experiment.trainer.signed_distance_maps(label.view_as(label)))
            self.assertIs(actual_depth,experiment.trainer.geometry_weights(label,affine,self.protocol['depth_geometry']))
            label[0,0,32,32,32]=0
            changed=experiment.trainer.signed_distance_maps(label)
            self.assertIsNot(changed,actual_sdf)
            self.assertTrue(torch.equal(changed,sdf(label)))
            self.assertEqual(fast.stats['sdf_misses'],2)

    def test_best_write_elision_preserves_latest_writes(self):
        original=experiment.base.save_checkpoint;calls=[]
        def counted(path,value):calls.append(path.name);original(path,value)
        experiment.base.save_checkpoint=counted
        try:
            with tempfile.TemporaryDirectory() as temp,RuntimeAcceleration(experiment,self.protocol,'cpu') as fast:
                folder=Path(temp);value={'model':{'weight':torch.arange(8)}}
                for _ in range(2):
                    experiment.base.save_checkpoint(folder/'checkpoint_best.pt',value)
                    experiment.base.save_checkpoint(folder/'checkpoint_latest.pt',value)
                self.assertEqual(calls.count('checkpoint_best.pt'),1)
                self.assertEqual(calls.count('checkpoint_latest.pt'),2)
                loaded=torch.load(folder/'checkpoint_latest.pt',weights_only=False)
                self.assertTrue(torch.equal(loaded['model']['weight'],value['model']['weight']))
        finally:experiment.base.save_checkpoint=original

    def test_augmented_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeAcceleration(experiment,{**self.protocol,'augmentation':'mild_v1'},'cpu')


if __name__=='__main__':unittest.main()
