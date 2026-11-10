"""Scientific parity, side gradients, budgets, scope and exact restart checks."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
import torch
from . import train, objectives as original
from .standard_resume import standard_protocol, PROTOCOL_SHA
from .asym32_matrix import extend_protocol as asym_protocol
from .winner_matrix import added_matrix, extend_protocol, retained_ids, scientific_key, ANCHOR
from .winner_objectives import components, regional_band
from .winner_resume import SOURCES
from .tests import target_for
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.native_branch_constraints.objectives import gradient_record
from scripts.depth_supervision.utils import seed_all

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'docs/experiments/band_focus_20261011_recovery/protocol.json'
BASE=asym_protocol(standard_protocol(json.loads(PROTOCOL.read_text())))
NEW=added_matrix(BASE['runs'])


class WinnerTests(unittest.TestCase):
    def setUp(self): seed_all(0)

    def test_scope_priority_dedup_and_frozen_sources(self):
        before=json.dumps(BASE,sort_keys=True)
        completed={'dice','bands',ANCHOR,'tile8_q25'}
        p=extend_protocol(BASE,completed)
        self.assertEqual(json.dumps(BASE,sort_keys=True),before)
        self.assertEqual(len(NEW),287)
        self.assertEqual(len(retained_ids(BASE['runs'])),21)
        self.assertEqual(p,json.loads(json.dumps(p)))
        active={e['id']:e for e in p['runs']}
        original_by_id={e['id']:e for e in BASE['runs']}
        for name in completed|set(retained_ids(BASE['runs'])):
            self.assertEqual(active[name],original_by_id[name])
            self.assertEqual(train.identity(p,active[name]),train.identity(BASE,original_by_id[name]))
        pending=[e['id'] for e in p['runs'] if e['id'] not in completed]
        asym_pending=[e['id'] for e in BASE['runs'] if e['id'].startswith('a32_') and e['id'] not in completed]
        self.assertEqual(pending[:len(asym_pending)],asym_pending)
        self.assertEqual(pending[len(asym_pending):len(asym_pending)+16],[e['id'] for e in NEW[:16]])
        self.assertNotIn('smooth_power1p0_mix50',active)
        self.assertNotIn('long_focus25',active)
        seen={scientific_key(e) for e in BASE['runs']}
        for e in NEW:
            self.assertNotIn(scientific_key(e),seen);seen.add(scientific_key(e))
            self.assertFalse(e.get('training'))
            self.assertEqual((e['fraction'],e['seed'],e['arm']),('5pct',0,'ordinary_swinunetr'))
        self.assertEqual(hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),PROTOCOL_SHA)
        for name,expected in BASE['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)

    def test_frozen_controls_and_equal_side_overrides_are_exact(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        for name in ('dice','bands','focus25',ANCHOR,'a32_neighbor_k3_qi40_qo25_mix75'):
            e=next(e for e in BASE['runs'] if e['id']==name)
            target=target_for(item,e)
            a=original.components(z,labels,e,target);b=components(z,labels,e,target)
            for k in a:
                torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
                torch.testing.assert_close(torch.autograd.grad(a[k],z,retain_graph=True)[0],
                    torch.autograd.grad(b[k],z,retain_graph=True)[0],atol=0,rtol=0)
            if e['band']['kind']=='regional':
                score=original.previous.foreground_score(z)
                band=dict(e['band'],inside_hard_mix=e['band']['hard_mix'],outside_hard_mix=e['band']['hard_mix'])
                torch.testing.assert_close(original.regional_band(score,e['band'],target),regional_band(score,band,target),atol=0,rtol=0)

    def test_every_new_objective_and_gradient_budget(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        for entry in NEW:
            values=components(z,labels,entry,target_for(item,entry))
            calibration=original.select_weights([gradient_record(values,z)],entry,BASE['calibration'])
            for value in values.values():
                self.assertTrue(torch.isfinite(value),entry['id'])
                self.assertTrue(torch.isfinite(torch.autograd.grad(value,z,retain_graph=True)[0]).all(),entry['id'])
            weights=calibration['coefficients']
            gd=torch.autograd.grad(values['dice'],z,retain_graph=True)[0]
            gb=torch.autograd.grad(weights['bands']*values['bands'],z,retain_graph=True)[0]
            gp=torch.autograd.grad(weights['pair']*values['pair'],z,retain_graph=True)[0]
            self.assertAlmostEqual(float(gb.norm()/gd.norm()),.1*entry['band_budget'],places=5)
            self.assertAlmostEqual(float(gp.norm()/(gd+gb).norm()),.1*entry['extra_budget'],places=5)

    def test_side_reduction_matches_explicit_topk_and_handles_empty_support(self):
        score=torch.linspace(-2,2,16,dtype=torch.double).reshape(2,2,4).requires_grad_()
        fg=torch.zeros_like(score,dtype=torch.bool);fg[0]=True
        target=dict(foreground=fg,valid=torch.ones_like(fg),distance=torch.ones_like(score))
        spec=dict(kind='regional',inner=3,outer=2,inside_weight=.4,penalty='brier',hard_fraction=.25,
            inside_fraction=.5,outside_fraction=.25,inside_hard_mix=.65,outside_hard_mix=.85)
        ei=(1-score[fg].sigmoid()).square();eo=score[~fg].sigmoid().square()
        expected=.4*(.35*ei.mean()+.65*ei.topk(4).values.mean())+.6*(.15*eo.mean()+.85*eo.topk(2).values.mean())
        actual=regional_band(score,spec,target)
        torch.testing.assert_close(actual,expected,atol=1e-15,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(actual,score,retain_graph=True)[0],
            torch.autograd.grad(expected,score)[0],atol=1e-15,rtol=0)
        for all_fg,valid in ((False,True),(True,True),(False,False)):
            t=dict(target,foreground=torch.full_like(fg,all_fg),valid=torch.full_like(fg,valid))
            v=regional_band(score,spec,t);g=torch.autograd.grad(v,score)[0]
            self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(g).all())
            if not valid:self.assertEqual(float(v),0.);self.assertEqual(float(g.abs().sum()),0.)
            elif all_fg:self.assertTrue((g<0).all())
            else:self.assertTrue((g>0).all())

    def test_inside_ce_changes_only_inside_feedback_and_strengthens_confident_misses(self):
        score=torch.full((5,5,5),-7.,dtype=torch.double,requires_grad=True)
        fg=torch.zeros_like(score,dtype=torch.bool);fg[1:4,1:4,1:4]=True
        valid=torch.ones_like(fg);valid[0]=False
        target=dict(foreground=fg,valid=valid,distance=torch.ones_like(score))
        spec=dict(inner=3,outer=2,inside_weight=.5,penalty='brier',hard_fraction=.6,hard_mix=.75,
                  inside_fraction=.6,outside_fraction=.15)
        a=regional_band(score,spec,target)
        b=regional_band(score,dict(spec,inside_penalty='hybrid',inside_bce_share=.1),target)
        ga=torch.autograd.grad(a,score)[0];gb=torch.autograd.grad(b,score)[0]
        torch.testing.assert_close(ga[~fg],gb[~fg],atol=0,rtol=0)
        self.assertGreater(float(gb[fg].abs().sum()),float(ga[fg].abs().sum())*10)
        self.assertEqual(float(gb[~valid].abs().sum()),0.)
        # Differentiability away from ranking ties on a mixed shell objective.
        score=torch.randn(5,5,5,dtype=torch.double,requires_grad=True)
        spec=dict(spec,inside_penalty='hybrid',inside_bce_share=.25,shells=[1,2,3],shell_weights=[.5,.35,.15])
        g=torch.autograd.grad(regional_band(score,spec,target),score)[0]
        idx=(2,2,2);eps=1e-5
        plus=score.detach().clone();minus=score.detach().clone();plus[idx]+=eps;minus[idx]-=eps
        numerical=(regional_band(plus,spec,target)-regional_band(minus,spec,target))/(2*eps)
        self.assertAlmostEqual(float(g[idx]),float(numerical),places=8)

    def test_exact_restart_for_independent_emphasis_and_inside_hybrid(self):
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=5,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        for index in (7,8):
            entry=NEW[index];name=entry['id']
            with tempfile.TemporaryDirectory() as directory,patch.object(train,'_load_pkl_dataframe',return_value=None),\
                 patch.object(train,'Patients',TinyPatients),patch.object(train,'model_factory',side_effect=tiny_model),\
                 patch.object(train,'objective_components',components):
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
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WinnerTests))
    if not result.wasSuccessful():raise SystemExit(1)
    value=dict(passed=True,tests=result.testsRun,new_objectives_checked=len(NEW),retained_originals=21,
        original_sources_unchanged=True,original_protocol_sha256=PROTOCOL_SHA,
        source_hashes={n:hashlib.sha256((ROOT/'scripts/band_focus'/n).read_bytes()).hexdigest() for n in SOURCES})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
