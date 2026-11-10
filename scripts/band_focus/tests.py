"""Control parity, anatomical edge cases, gradient and restart gates."""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.band_evolution.matrix import experiment_matrix as prior_matrix
from scripts.band_evolution.objectives import components as prior_components, select_weights as prior_weights
from scripts.band_evolution.geometry import Targets as PriorTargets, move
from scripts.depth_supervision.utils import atomic_json, digest, seed_all
from scripts.depth_supervision.losses import combined_loss
from scripts.native_branch_constraints.objectives import gradient_record
from .matrix import experiment_matrix, focus
from .objectives import components, select_weights, regional_band, reduce_error, neighborhood_rank, box_sum, tile_focus, at_epoch
from .geometry import Targets
from . import train


def target_for(item, entry, device='cpu'):
    p=dict(native_geometry={item['case_name']:dict(native_shape=list(item['label'].shape[1:]))})
    return Targets(Path('.'),p,entry).get(item,device)


class ScientificTests(unittest.TestCase):
    def setUp(self):
        seed_all(0)

    def test_frozen_matrix_json_roundtrip(self):
        entries=experiment_matrix()
        self.assertEqual(json.loads(json.dumps(entries)),entries)
        original=Path(__file__).resolve().parents[2]/'docs/experiments/band_focus_20261011/protocol.json'
        if original.exists():
            self.assertEqual(json.loads(original.read_text())['runs'],entries)

    def test_exact_control_values_gradients_and_calibration(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        old={e['id']:e for e in prior_matrix()};new={e['id']:e for e in experiment_matrix()}
        names={'bands':'bands','independent13':'independent13','focus25':'band_focus_top25_in50','strong_inside':'budget_protect_inside_1p5'}
        for name,previous in names.items():
            a=components(z,labels,new[name],target_for(item,new[name]))
            b=prior_components(z,labels,old[previous],target_for(item,old[previous]))
            for key in a:
                torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
                torch.testing.assert_close(torch.autograd.grad(a[key],z,retain_graph=True)[0],
                    torch.autograd.grad(b[key],z,retain_graph=True)[0],atol=0,rtol=0)
            settings=dict(prefix_epochs=5,target_ratio=.1,cap_ratio=.5)
            ca=select_weights([gradient_record(a,z)],new[name],settings)['coefficients']
            cb=prior_weights([gradient_record(b,z)],old[previous],settings)['coefficients']
            for key in ca:self.assertAlmostEqual(ca[key],cb[key],places=12)

    def test_hard_fraction_and_mix_have_explicit_mass(self):
        e=torch.tensor([1.,2.,3.,4.],requires_grad=True)
        loss=reduce_error(e,torch.ones_like(e),dict(hard_fraction=.25,hard_mix=.5))
        self.assertEqual(float(loss),3.25)
        torch.testing.assert_close(torch.autograd.grad(loss,e)[0],torch.tensor([.125,.125,.125,.625]))
        self.assertEqual(float(reduce_error(e,torch.ones_like(e),dict(hard_fraction=1.,hard_mix=.7))),2.5)

    def test_box_and_neighborhood_do_not_cancel_or_include_padding(self):
        x=torch.arange(9**3,dtype=torch.float32).reshape(9,9,9)
        expected=torch.nn.functional.avg_pool3d(x[None,None],3,stride=1,padding=1,count_include_pad=True)[0,0]*27
        torch.testing.assert_close(box_sum(x,3),expected,atol=.02,rtol=1e-6)
        mask=torch.zeros_like(x,dtype=torch.bool);mask[3:6,3:6,3:6]=True
        errors=torch.ones_like(x);errors[~mask]=1e6
        rank=neighborhood_rank(errors,mask,3)
        torch.testing.assert_close(rank[mask],torch.ones_like(rank[mask]))
        self.assertFalse(rank.requires_grad)

    def test_tile_one_fraction_equals_mean_with_unequal_counts(self):
        e=torch.rand(9,11,7,requires_grad=True);m=torch.rand_like(e)>.6;w=torch.rand_like(e)+.1
        loss=tile_focus(e,m,w,dict(tile=4,hard_fraction=1.,hard_mix=.5))
        expected=(e[m]*w[m]).sum()/w[m].sum()
        torch.testing.assert_close(loss,expected)
        g=torch.autograd.grad(loss,e)[0]
        self.assertTrue((g[~m]==0).all());self.assertTrue((g[m]>0).all())

    def test_perfect_empty_full_and_padding(self):
        entries=[e for e in experiment_matrix() if e['family'] in ('neighborhood','spatial_coverage','smooth_focus','ap_focus')]
        for value in (0,1):
            fg=torch.full((9,9,9),bool(value));valid=torch.zeros_like(fg);valid[1:8,1:8,1:8]=True
            target=dict(foreground=fg,valid=valid,distance=torch.ones(9,9,9))
            score=torch.full((9,9,9),15. if value else -15.,requires_grad=True)
            for e in entries[::3]:
                loss=regional_band(score,e['band'],target)
                self.assertTrue(torch.isfinite(loss));self.assertLess(float(loss),1e-10)
                g=torch.autograd.grad(loss,score,retain_graph=True)[0]
                self.assertTrue((g[~valid]==0).all())
            target['valid'].zero_()
            self.assertEqual(float(regional_band(score,focus(),target)),0.)

    def test_neighborhood_gradient_signs_and_finite_difference(self):
        fg=torch.zeros(7,7,7,dtype=torch.bool);fg[2:5,2:5,2:5]=True
        valid=torch.ones_like(fg);valid[0]=False
        target=dict(foreground=fg,valid=valid,distance=torch.ones(7,7,7,dtype=torch.double))
        score=torch.randn(7,7,7,dtype=torch.double,requires_grad=True)
        spec=dict(focus(),neighborhood=[3],point_rank_share=.5)
        loss=regional_band(score,spec,target);g=torch.autograd.grad(loss,score)[0]
        self.assertTrue((g[fg&valid]<0).all());self.assertTrue((g[~fg&valid]>0).all())
        # Hard ranks are constant away from ties. Perturb a single score locally.
        idx=(3,3,3);eps=1e-5
        plus=score.detach().clone();minus=score.detach().clone();plus[idx]+=eps;minus[idx]-=eps
        numerical=(regional_band(plus,spec,target)-regional_band(minus,spec,target))/(2*eps)
        self.assertAlmostEqual(float(g[idx]),float(numerical),places=7)

    def test_ap_conditional_targets_and_background_gradient(self):
        e=next(e for e in experiment_matrix() if e['id']=='ap_bce_w3_share10')
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        values=components(z,labels,e,target_for(item,e));g=torch.autograd.grad(values['ap'],z)[0]
        self.assertTrue((g[:,0]==0).all())
        self.assertTrue((g[0,:,labels[0,0]==0]==0).all())
        self.assertGreater(float(g.abs().sum()),0.)

    def test_curriculum_is_pure_and_reaches_exact_final(self):
        e=next(e for e in experiment_matrix() if e['id']=='curriculum_wide_gentle_e20')
        before=json.dumps(e,sort_keys=True)
        self.assertEqual(at_epoch(e,5)['band']['hard_fraction'],.75)
        self.assertEqual(at_epoch(e,5)['band']['hard_mix'],0.)
        self.assertEqual(at_epoch(e,20)['band'],e['band'])
        self.assertEqual(json.dumps(e,sort_keys=True),before)

    def test_all_objectives_finite_and_budgeted(self):
        item=synthetic_item();labels=item['label'][None];z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        entries=experiment_matrix();self.assertEqual(len(entries),193)
        settings=dict(prefix_epochs=5,target_ratio=.1,cap_ratio=.5)
        for entry in entries:
            values=components(z,labels,entry,target_for(item,entry))
            result=select_weights([gradient_record(values,z)],entry,settings)
            for key,value in values.items():
                self.assertTrue(torch.isfinite(value),entry['id'])
                self.assertTrue(torch.isfinite(torch.autograd.grad(value,z,retain_graph=True)[0]).all(),entry['id'])
            if 'bands' not in values:continue
            weights=result['coefficients'];gd=torch.autograd.grad(values['dice'],z,retain_graph=True)[0]
            gb=torch.autograd.grad(weights['bands']*values['bands'],z,retain_graph=True)[0]
            self.assertAlmostEqual(float(gb.norm()/gd.norm()),.1*entry['band_budget'],places=5)
            extras=[k for k in values if k not in ('dice','bands')]
            if extras:
                ge=torch.autograd.grad(sum(weights[k]*values[k] for k in extras),z,retain_graph=True)[0]
                self.assertAlmostEqual(float(ge.norm()/(gd+gb).norm()),.1*entry['extra_budget'],places=5)

    def test_nested_transfer(self):
        e=next(e for e in experiment_matrix() if e['id']=='focus25')
        target=target_for(synthetic_item(),e)
        moved=move(target,'meta')
        for group in moved['pairs']:
            for value in group.values():
                self.assertTrue(all(t.device.type=='meta' for t in value.values()))

    def test_restart_curriculum_extended_and_dice(self):
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=4,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        entries={e['id']:e for e in experiment_matrix()}
        for name in ('dice','curriculum_wide_gentle_e20','long_focus25'):
            entry=entries[name]
            if entry.get('training'):
                entry=dict(entry,training=dict(max_epochs=5,step_size=4,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.)))
            with tempfile.TemporaryDirectory() as directory,patch.object(train,'_load_pkl_dataframe',return_value=None),\
                 patch.object(train,'Patients',TinyPatients),patch.object(train,'model_factory',side_effect=tiny_model):
                root=Path(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    expected=train.run(p,root/'whole',entry,device_override='cpu')
                    train.run(p,root/'resume',entry,stop_after=3,device_override='cpu')
                    actual=train.run(p,root/'resume',entry,device_override='cpu')
                self.assertEqual(expected['validation'],actual['validation'])
                self.assertEqual(expected['completed_epochs'],5 if name=='long_focus25' else 4)
                a=torch.load(root/'whole/runs'/name/'best.pt',weights_only=False)
                b=torch.load(root/'resume/runs'/name/'best.pt',weights_only=False)
                for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)


def model_smoke(device_name, root, pkl_override=None):
    seed_all(0);device=torch.device(device_name);p=json.loads((root/'protocol.json').read_text())
    dataset=pkl_override or p['pkl']
    if digest(dataset)!=p['pkl_sha256']:
        from scripts.band_chamfer.audit_ap import audit
        actual,_=audit(p,dataset)
        expected=json.loads((root/'ap_training_geometry_cluster.json').read_text())
        if actual!=expected:raise ValueError('Training-array fingerprints differ')
    patients=train.Patients(train._load_pkl_dataframe(dataset),p['splits']['5pct']['train'],p['native_geometry'])
    model=train.model_factory(device)
    if (root/'shared_warmup/latest.pt').exists():
        state=torch.load(root/'shared_warmup/latest.pt',map_location=device,weights_only=False)
        model.load_state_dict(state['model']);del state
    assert sum(q.numel() for q in model.parameters())==p['model_parameters']
    item=patients[0];model.eval()
    with train.autocast(device):logits=model(item['image'][None].to(device))
    detached=logits.detach().float().requires_grad_(True);labels=item['label'][None].to(device)
    checks={};geometry={};timings={}
    previous_entries={e['id']:e for e in prior_matrix()}
    for entry in p['runs']:
        started=time.monotonic();cache=Targets(root,p,entry)
        for other in patients.items:
            target=cache.get(other,'cpu')
            assert len(target['gt'])
        values=components(detached,labels,entry,cache.get(item,device))
        record=gradient_record(values,detached)
        reference=p.get('reference_controls',{}).get(entry['id'])
        if reference:
            prior_entry=previous_entries[reference]
            prior_values=prior_components(detached,labels,prior_entry,PriorTargets(root,p,prior_entry).get(item,device))
            for key,value in values.items():
                torch.testing.assert_close(value,prior_values[key],rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(torch.autograd.grad(value,detached,retain_graph=True)[0],
                    torch.autograd.grad(prior_values[key],detached,retain_graph=True)[0],rtol=1e-5,atol=1e-8)
        calibrated=select_weights([record],entry,p['calibration'])
        assert all(math.isfinite(v) for v in record['losses'].values()),entry['id']
        checks[entry['id']]=dict(losses=record['losses'],coefficients=calibrated['coefficients'])
        geometry[entry['id']]=cache.statistics();timings[entry['id']]=time.monotonic()-started
        print(json.dumps(dict(stage='smoke_objective',id=entry['id'],seconds=timings[entry['id']])),flush=True)
    # One real-model update exercises spatial selection and conditional A/P gradients.
    entry=next(e for e in p['runs'] if e['id']=='ap_bce_w3_share10')
    entry=dict(entry,band=dict(entry['band'],neighborhood=[3],point_rank_share=.5))
    values=components(logits,labels,entry,Targets(root,p,entry).get(item,device))
    coefficients=select_weights([gradient_record(values,logits)],entry,p['calibration'])['coefficients']
    loss=combined_loss(values,coefficients)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False);loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    watched={n:(q,q.detach().clone()) for n,q in model.named_parameters() if n in ('swinViT.patch_embed.proj.weight','out.conv.conv.weight')}
    optimizer.step();assert all(not torch.equal(q,before) for q,before in watched.values())
    return dict(passed=True,objectives_checked=len(checks),real_training_cases_checked=len(patients),
                device=device_name,parameters_updated=True,ordinary_swinunetr=True,
                spatial_ap_model_update=True,objective_checks=checks,geometry=geometry,timings=timings)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('unit','model'),required=True)
    parser.add_argument('--root',type=Path);parser.add_argument('--device',default='cpu');parser.add_argument('--pkl')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=='unit':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificTests))
        if not result.wasSuccessful():raise SystemExit(1)
        value=dict(passed=True,tests=result.testsRun)
    else:value=model_smoke(args.device,args.root,args.pkl)
    from .prepare import ROOT,sha
    value['tested_sources']={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'scripts/band_focus').glob('*') if f.suffix in ('.py','.sh')}
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('tested_sources','geometry','objective_checks','timings')}))


if __name__=='__main__':main()
