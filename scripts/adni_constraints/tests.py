"""Binary loss oracles and a real no-augmentation calibration/training smoke."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.adni_constraints import objectives as obj
from scripts.adni_constraints.train import configure, trainer
from scripts.adni_baselines.core import ROOT, base, PreparedADNI, MODELS
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss


class BinaryObjectives(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.target = torch.zeros(1,1,64,64,64,dtype=torch.long)
        self.target[:,:,25:38,19:45,26:39] = 1
        self.logits = torch.randn(1,2,64,64,64,requires_grad=True)

    def test_grouped_foreground_equivalence(self):
        three = torch.cat([self.logits[:,:1], self.logits[:,1:]-np.log(2), self.logits[:,1:]-np.log(2)],1)
        for binary, original in ((obj.BinaryEdge(), BoundaryEdgeConsistencyLoss()),
                                 (obj.bands(), OuterBoundaryBandLoss())):
            left, right = binary(self.logits,self.target), original(three,self.target)
            left = left.loss if hasattr(left,'loss') else left
            right = right.loss if hasattr(right,'loss') else right
            self.assertTrue(torch.allclose(left,right,atol=2e-7))
            g1 = torch.autograd.grad(left,self.logits,retain_graph=True)[0]
            g2 = torch.autograd.grad(right,self.logits,retain_graph=True)[0]
            self.assertTrue(torch.allclose(g1,g2,atol=2e-8))

    def test_distance_direction_and_depth_normalization(self):
        sdf = obj.signed_distance_maps(self.target)
        self.assertTrue((sdf[self.target.bool()] < 0).all())
        self.assertTrue((sdf[~self.target.bool()] > 0).all())
        loss = obj.sdf_loss(self.logits,self.target,sdf)
        gradient = torch.autograd.grad(loss,self.logits)[0][:,1:]
        self.assertTrue((gradient[self.target.bool()] < 0).all())
        self.assertTrue((gradient[~self.target.bool()] > 0).all())
        weights = obj.geometry_weights(self.target,np.eye(4),obj.GEOMETRY)
        self.assertAlmostEqual(float(weights.sum()),1.,places=6)
        uniform = torch.zeros_like(self.logits)
        self.assertAlmostEqual(float(obj.depth_loss(uniform,self.target,weights)),np.log(2),places=6)


def smoke(output):
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    # Any accidental call to augmentation fails the actual optimizer loop.
    def forbidden(*args, **kwargs):
        raise AssertionError('User requires unaugmented training')
    base.augment_mild = forbidden
    data = ROOT/'datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'
    for identity in ('basic_unet16','nnunet'):
        parent_dir = ROOT/'experiments/adni_baselines_5pct_fold0_seed0_20261013/protocols'/identity
        protocol = json.loads((parent_dir/'protocol.json').read_text())
        alias = MODELS[identity]['alias']
        schedule = dict(max_steps=3,prefix_updates=1,sdf_prefix_updates=2,baseline_flops_per_update=100,
                        early_flops_per_update=110,flops_per_update=120,calibration_flops=50,
                        sdf_warmup_updates=1,constraint_warmup_updates=1,budget_scope='optimization_only')
        arm = 'dice_bands_edge_sdf_depth'
        split = json.loads((parent_dir/'splits_10patients.json').read_text())[0]
        protocol.update(augmentation='none',trainer_alias=alias,depth_geometry=obj.GEOMETRY,
            native_geometry={c:dict(model_affine=np.eye(4).tolist()) for c in split['train']+split['val']},
            budget_scope='optimization_only',sdf_policy={arm:'delayed_5_of_9'},constraint_warmup_updates=1,
            models={alias:dict(parameters=protocol['models'][alias]['parameters'],objectives={arm:schedule})})
        directory=output/identity/'protocol'
        directory.mkdir(parents=True)
        for name in ('splits_10patients.json','nnunet_architecture.json'):
            (directory/name).write_bytes((parent_dir/name).read_bytes())
        base.atomic_json(directory/'protocol.json',protocol)
        configure(protocol,arm)
        args=SimpleNamespace(model=alias,objective=arm,fold=0,seed=0,pkl=data,
             output=output/identity/'run',protocol_dir=directory,device='cpu',threads=2,resume=False,smoke_steps=3)
        trainer.run(args)
        done=json.loads((args.output/'completion.json').read_text())
        calibration=json.loads((args.output/'calibration.json').read_text())
        if done['updates']!=3 or done['calibration_attempts']!=1 or set(calibration['weights'])!={'bands','edge','sdf','depth'}:
            raise AssertionError('All four constraints must execute after training-only calibration')
        if [r['case_name'] for r in calibration['cases']]!=split['train']:
            raise AssertionError('Calibration population differs')
        reports.append(dict(model=identity,status='passed',updates=3,augmented_calls=0,
                            calibrated_terms=list(calibration['weights'])))
    base.atomic_json(output/'smoke_summary.json',dict(status='passed',runs=reports,
        warning='Smoke uses artificial tiny accounting and shortened ramps; never production results'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(BinaryObjectives))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.smoke_output:
        smoke(args.smoke_output)
