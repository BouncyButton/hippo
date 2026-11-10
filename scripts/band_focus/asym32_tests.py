"""Scope, gradients, side support and exact restart tests for the extension."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .asym32_matrix import added_matrix, extend_protocol, scientific_key
from .standard_resume import standard_protocol, PROTOCOL_SHA
from .tests import target_for
from .objectives import components, select_weights, regional_band
from . import train
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.native_branch_constraints.objectives import gradient_record
from scripts.depth_supervision.utils import seed_all

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/experiments/band_focus_20261011_recovery/protocol.json'
BASE=standard_protocol(json.loads(PROTOCOL.read_text()))
NEW=added_matrix(BASE['runs'])


class ExtensionTests(unittest.TestCase):
    def setUp(self):seed_all(0)

    def test_frozen_subset_priority_and_no_equivalent_duplicates(self):
        original=json.dumps(BASE,sort_keys=True)
        p=extend_protocol(BASE)
        self.assertEqual(json.dumps(BASE,sort_keys=True),original)
        self.assertEqual(p['runs'][len(NEW):],BASE['runs'])
        self.assertEqual(len(NEW),172)
        self.assertEqual(len(p['runs']),362)
        self.assertEqual(json.loads(json.dumps(p)),p)
        self.assertEqual(p['runs'][0]['id'],'a32_hard_q35_mix75')
        seen={scientific_key(e) for e in BASE['runs']}
        for e in NEW:
            key=scientific_key(e);self.assertNotIn(key,seen);seen.add(key)
            self.assertEqual((e['band']['inner'],e['band']['outer']),(3,2))
            self.assertFalse(e.get('training'))
            self.assertEqual((e['fraction'],e['seed'],e['arm']),('5pct',0,'ordinary_swinunetr'))
        for e in BASE['runs']:
            self.assertEqual(train.identity(p,e),train.identity(BASE,e))
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),PROTOCOL_SHA)
        for name,expected in BASE['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)

    def test_every_added_objective_finite_with_correct_gradient_budgets(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        for entry in NEW:
            values=components(z,labels,entry,target_for(item,entry))
            record=gradient_record(values,z)
            calibration=select_weights([record],entry,BASE['calibration'])
            for key,value in values.items():
                self.assertTrue(torch.isfinite(value),entry['id'])
                self.assertTrue(torch.isfinite(torch.autograd.grad(value,z,retain_graph=True)[0]).all(),entry['id'])
            weights=calibration['coefficients']
            gd=torch.autograd.grad(values['dice'],z,retain_graph=True)[0]
            gb=torch.autograd.grad(weights['bands']*values['bands'],z,retain_graph=True)[0]
            self.assertAlmostEqual(float(gb.norm()/gd.norm()),.1*entry['band_budget'],places=5)
            if entry['pair']:
                ge=torch.autograd.grad(weights['pair']*values['pair'],z,retain_graph=True)[0]
                self.assertAlmostEqual(float(ge.norm()/(gd+gb).norm()),.1*entry['extra_budget'],places=5)

    def test_asymmetric_support_and_neighborhood_side_gradients(self):
        shape=(9,9,9)
        foreground=torch.zeros(shape,dtype=torch.bool);foreground[2:7,2:7,2:7]=True
        valid=torch.ones(shape,dtype=torch.bool);valid[0]=False
        distance=torch.arange(9,dtype=torch.double)[:,None,None].expand(shape)/2
        score=torch.randn(shape,dtype=torch.double,requires_grad=True)
        target=dict(foreground=foreground,valid=valid,distance=distance)
        entry=next(e for e in NEW if e['id']=='a32_neighbor_k3_qi25_qo40_mix75')
        loss=regional_band(score,entry['band'],target)
        grad=torch.autograd.grad(loss,score)[0]
        inside=foreground&valid&(distance>0)&(distance<=3)
        outside=(~foreground)&valid&(distance>0)&(distance<=2)
        self.assertTrue((grad[inside]<0).all())
        self.assertTrue((grad[outside]>0).all())
        self.assertTrue((grad[~(inside|outside)]==0).all())
        idx=(3,3,3);eps=1e-5
        plus=score.detach().clone();minus=score.detach().clone();plus[idx]+=eps;minus[idx]-=eps
        numerical=(regional_band(plus,entry['band'],target)-regional_band(minus,entry['band'],target))/(2*eps)
        self.assertAlmostEqual(float(grad[idx]),float(numerical),places=7)

    def test_exact_restart_for_added_focus_and_mixed_neighborhood(self):
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=5,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        for name in ('a32_hard_q35_mix75','a32_neighbor_k3_qi25_qo40_mix75'):
            entry=next(e for e in NEW if e['id']==name)
            with tempfile.TemporaryDirectory() as directory,patch.object(train,'_load_pkl_dataframe',return_value=None),\
                 patch.object(train,'Patients',TinyPatients),patch.object(train,'model_factory',side_effect=tiny_model):
                root=Path(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    expected=train.run(p,root/'whole',entry,device_override='cpu')
                    train.run(p,root/'resume',entry,stop_after=3,device_override='cpu')
                    actual=train.run(p,root/'resume',entry,device_override='cpu')
                self.assertEqual(expected['validation'],actual['validation'])
                a=torch.load(root/'whole/runs'/name/'best.pt',weights_only=False)
                b=torch.load(root/'resume/runs'/name/'best.pt',weights_only=False)
                for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionTests))
    if not result.wasSuccessful():raise SystemExit(1)
    names=('asym32_matrix.py','asym32_resume.py','asym32_tests.py','run_asym32.sh')
    value=dict(passed=True,tests=result.testsRun,new_objectives_checked=len(NEW),total_objectives=362,
        original_sources_unchanged=True,original_protocol_sha256=PROTOCOL_SHA,
        source_hashes={n:hashlib.sha256((ROOT/'scripts/band_focus'/n).read_bytes()).hexdigest() for n in names})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
