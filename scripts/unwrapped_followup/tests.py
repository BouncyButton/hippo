"""Check loss isolation, gate semantics, gradients, and exact restart behavior."""
import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.depth_supervision.losses import components, combined_loss
from scripts.depth_supervision.train import autocast
from scripts.depth_supervision.utils import atomic_json, digest
from .models import FollowupSwin, uncertainty_gate
from .prepare import entries
from . import train


class TinyModel(torch.nn.Module):
    def __init__(self, device, arm, gate):
        super().__init__();self.native=torch.nn.Conv3d(1,3,1).to(device)
        self.encoder=None;self.settings={}
    def forward(self,x,atlas=None):return self.native(x)


class TinyPatients:
    def __init__(self,frame,names,geometry):
        self.items=[]
        for i,name in enumerate(names):
            image=torch.arange(64,dtype=torch.float32).reshape(1,4,4,4)/64+i/10
            label=(torch.arange(64).reshape(1,4,4,4)+i)%3
            self.items.append(dict(case_name=name,image=image,label=label,affine=np.eye(4)))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


class TinyGeometry:
    def __init__(self,*args):pass
    def get(self,item,fields=False):
        return dict(weights=np.ones((4,4,4),np.float32)/64,bin_ids=np.zeros((4,4,4),int))


class FollowupTests(unittest.TestCase):
    def test_registered_questions_and_counts(self):
        runs,pairs=entries()
        self.assertEqual(len(runs),24)
        self.assertEqual(len({r['id'] for r in runs}),24)
        self.assertEqual(sum(r['group']=='dice_ablation' for r in runs),4)
        self.assertEqual(sum(r['group']=='replication' for r in runs),12)
        self.assertEqual(sum(r['group']=='uncertainty' for r in runs),8)
        self.assertTrue(all(r['objective']=='dice' for r in runs if r['group']=='dice_ablation'))
        self.assertEqual(pairs['5pct'],['plane_position','unfold_position'])

    def test_gate_uncertainty_and_detachment(self):
        logits=torch.tensor([[[[[40.,0.]]]],[[[[0.,0.]]]],[[[[0.,0.]]]]],requires_grad=True).reshape(1,3,1,1,2)
        gate=uncertainty_gate(logits,'entropy')
        self.assertLess(gate.flatten()[0].item(),1e-8)
        self.assertAlmostEqual(gate.flatten()[1].item(),1.,places=6)
        self.assertFalse(gate.requires_grad)
        mean=uncertainty_gate(logits,'mean_entropy')
        torch.testing.assert_close(mean,torch.ones_like(mean)*gate.mean())
        self.assertAlmostEqual(mean.mean().item(),gate.mean().item(),places=7)
        with self.assertRaises(ValueError):uncertainty_gate(logits,'invalid')

    def test_dice_has_no_depth_component(self):
        logits=torch.randn(1,3,4,4,4,requires_grad=True);labels=torch.zeros(1,1,4,4,4,dtype=torch.long)
        parts=components(logits,labels,None,'dice')
        self.assertEqual(set(parts),{'dice'})
        torch.testing.assert_close(combined_loss(parts,{'depth':100.},1.),parts['dice'])

    def test_restart_equivalence_and_completion_skip(self):
        p=dict(identity_sha='test',atlas_protocol_sha256='prior',splits={'5pct':dict(train=['a','b'],val=['v'])},
               native_geometry={},pkl='unused',learning_rate=.01,weight_decay=1e-5,step_size=2,lr_gamma=.5,
               geometry={},fixed_depth_coefficients={'5pct':.1},max_epochs=7,
               early_stopping=dict(patience=8,min_epochs=25,min_delta=.0005))
        entry=dict(id='resume_test',seed=2,fraction='5pct',arm='depth',objective='depth',gate='none',group='test')
        with tempfile.TemporaryDirectory() as directory, patch.object(train,'FollowupSwin',TinyModel), \
             patch.object(train,'Patients',TinyPatients),patch.object(train,'GeometryCache',TinyGeometry), \
             patch.object(train,'_load_pkl_dataframe',return_value=None):
            a,b=Path(directory)/'continuous',Path(directory)/'resumed'
            train.run(p,a,entry,device_override='cpu')
            train.run(p,b,entry,stop_after=6,device_override='cpu')
            bad=dict(entry,seed=1)
            with self.assertRaisesRegex(ValueError,'Resume identity'):
                train.run(p,b,bad,device_override='cpu')
            # Recover a missing best publication when the current state is best.
            state=torch.load(b/'runs/resume_test/latest.pt',weights_only=False)
            if state['best_epoch']==6:(b/'runs/resume_test/best.pt').unlink()
            rb=train.run(p,b,entry,device_override='cpu')
            sa=torch.load(a/'runs/resume_test/best.pt',weights_only=False)
            sb=torch.load(b/'runs/resume_test/best.pt',weights_only=False)
            self.assertEqual(sa['epoch'],sb['epoch'])
            for k in sa['model']:torch.testing.assert_close(sa['model'][k],sb['model'][k],rtol=0,atol=0)
            self.assertEqual(rb['optimizer_updates'],14)
            with (a/'runs/resume_test/metrics.csv').open() as fa, (b/'runs/resume_test/metrics.csv').open() as fb:
                ha,hb=list(csv.DictReader(fa)),list(csv.DictReader(fb))
            for ra,rb_row in zip(ha,hb):
                self.assertEqual({k:v for k,v in ra.items() if k!='elapsed_seconds'},
                                 {k:v for k,v in rb_row.items() if k!='elapsed_seconds'})
            with patch.object(train,'FollowupSwin',side_effect=AssertionError('Completed run retrained')):
                self.assertEqual(train.run(p,b,entry,device_override='cpu'),rb)


def model_smoke(atlas,device_name='cpu',arm='unfold_attention'):
    torch.set_num_threads(4);torch.manual_seed(0)
    device=torch.device(device_name)
    model=FollowupSwin(device,arm,'none')
    x=torch.randn(1,1,64,64,64,device=device)
    model.eval()
    with torch.no_grad(),autocast(device):
        original=model(x,atlas)
        model.gate_mode='entropy'
        unit=model(x,atlas,unit_gate=True)
        torch.testing.assert_close(original,unit,rtol=0,atol=0)
    checks=[]
    labels=(torch.arange(64,device=device)[None,None,:,None,None].expand(1,1,64,64,64)//22).clamp_max(2)
    for mode in ('entropy','mean_entropy'):
        model.train();model.gate_mode=mode;model.zero_grad(set_to_none=True)
        with autocast(device):logits=model(x,atlas)
        loss=components(logits,labels,None,'dice')['dice'];loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        for module in (model.native,model.encoder,model.fusion):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
        checks.append(dict(mode=mode,loss=float(loss.detach())))
    return dict(passed=True,device=device_name,arm=arm,unit_gate_exactly_matches_prior=True,
                all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                parameter_count=sum(p.numel() for p in model.parameters()),
                native_encoder_fusion_gradients_nonzero=True,checks=checks,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if device_name=='cuda' else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path)
    parser.add_argument('--smoke-atlas',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--cuda-fraction',choices=['5pct','full'])
    args=parser.parse_args();torch.set_num_threads(2)
    if args.cuda_fraction:
        p=json.loads((args.root/'protocol.json').read_text())
        manifest=json.loads((args.root/'atlas'/args.cuda_fraction/'manifest.json').read_text())
        largest=max(manifest['reports'],key=lambda r:sum(np.prod(a['grid_shape']) for a in r['charts_audit'] if a['view']=='unfold'))
        atlas_path=args.root/'atlas'/args.cuda_fraction/(largest['case']+'.npz')
        with np.load(atlas_path) as z:atlas={k:z[k] for k in z.files}
        result=model_smoke(atlas,'cuda',p['pairs'][args.cuda_fraction][1]);result['case']=largest['case']
        atomic_json(args.root/f'cuda_preflight_{args.cuda_fraction}.json',result)
    elif args.smoke_atlas:
        with np.load(args.smoke_atlas) as z:atlas={k:z[k] for k in z.files}
        result=model_smoke(atlas)
    else:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(FollowupTests)
        result_=unittest.TextTestRunner(verbosity=2).run(suite)
        result=dict(passed=result_.wasSuccessful(),tests=result_.testsRun)
        if not result['passed']:raise SystemExit(1)
    if args.output:atomic_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
