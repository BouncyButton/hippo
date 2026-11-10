"""A/P geometry, conditional gradients, frozen backgrounds, budgets and restart."""
import argparse
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import ap82_matrix as matrix, train
from .ap82_geometry import Targets, build
from .ap82_objectives import components, select_weights, install
from .ap82_resume import SOURCES
from .winner_objectives import components as base_components
from .winner_tests import BASE, NEW
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.native_branch_constraints.objectives import gradient_record
from scripts.depth_supervision.losses import combined_loss
from scripts.depth_supervision.utils import seed_all, save_checkpoint

ROOT=Path(__file__).resolve().parents[2]
BASES=dict(dice=next(e for e in BASE['runs'] if e['id']=='dice'),winner=next(e for e in NEW if e['id']==matrix.ANCHOR))
COEFFICIENTS=dict(dice={},winner=dict(bands=.032323425078637294,pair=.008503534232521521))
FIRST=matrix.stage1(BASES,COEFFICIENTS)
SECOND=matrix.stage2(BASES,COEFFICIENTS,FIRST[0])


def target(item,e):
    p=dict(native_geometry={item['case_name']:dict(native_shape=list(item['label'].shape[1:]))})
    return Targets(Path('.'),p,e).get(item,'cpu')


@contextlib.contextmanager
def installed():
    attrs=('objective_components','select_weights','Targets','TERM_ORDER','write_history')
    with contextlib.ExitStack() as stack:
        for name in attrs:stack.enter_context(patch.object(train,name,getattr(train,name)))
        install();yield


class APTests(unittest.TestCase):
    def setUp(self):seed_all(0)

    def test_matrix_selection_mirroring_and_bounded_interactions(self):
        self.assertEqual(len(matrix.retained_ids()),38)
        self.assertEqual((len(FIRST),len(SECOND)),(30,44))
        rows=[dict(objective=e['id'],macro_dice=.8,hd95_mm=2.) for e in FIRST]
        rows[0]['macro_dice']=.81
        parent=matrix.select_parent(FIRST,rows);self.assertEqual(parent,FIRST[0])
        rows += [dict(objective=e['id'],macro_dice=.8101,hd95_mm=2.) for e in SECOND]
        third,factors=matrix.stage3(BASES,COEFFICIENTS,parent,SECOND,rows)
        self.assertEqual(third,[])
        for row in rows[30:]:row['macro_dice']=.811
        third,factors=matrix.stage3(BASES,COEFFICIENTS,parent,SECOND,rows)
        self.assertEqual(len(third),32);self.assertEqual(len(factors),5)
        entries=FIRST+SECOND+third
        self.assertEqual(len({matrix.scientific_key(e) for e in entries}),len(entries))
        for a,b in zip(entries[::2],entries[1::2]):
            self.assertEqual((a['background'],b['background']),matrix.BACKGROUNDS)
            self.assertEqual(a['ap82'],b['ap82'])
            self.assertFalse(a.get('training'));self.assertFalse(b.get('training'))
        self.assertEqual(entries,json.loads(json.dumps(entries)))

    def test_geometry_actual_mixed_interface_and_empty_cases(self):
        y=np.zeros((7,9,7),int);y[1:6,1:5,1:6]=2;y[1:6,5:8,1:6]=1
        y[2:4,4,2:4]=1 # A mixed slice must remain represented.
        valid=np.ones_like(y,bool);valid[0]=False
        t=build(y,valid)
        self.assertTrue(torch.equal(t['valid'],torch.tensor((y>0)&valid)))
        self.assertEqual(float(t['distance'][2,4,2]),.5)
        self.assertEqual(float(t['distance'][1,4,1]),.5)
        for d,g in zip((1,3),t['pairs']):
            i,j=[g[s]['index'].numpy() for s in ('inside','outside')]
            self.assertTrue((y.ravel()[i]==1).all());self.assertTrue((y.ravel()[j]==2).all())
            a=np.column_stack(np.unravel_index(i,y.shape));b=np.column_stack(np.unravel_index(j,y.shape))
            np.testing.assert_allclose(np.linalg.norm(a-b,axis=1),d)
            for fraction in np.linspace(0,1,d+1):
                xyz=np.rint(a+(b-a)*fraction).astype(int)
                self.assertTrue((y[tuple(xyz.T)]>0).all())
        for value in (0,1,2):
            empty=build(np.full_like(y,value),valid)
            self.assertTrue(torch.isinf(empty['distance']).all())
            self.assertTrue(all(len(g['inside']['index'])==0 for g in empty['pairs']))

    def test_conditional_signs_background_exclusion_and_endpoint_independence(self):
        item=synthetic_item();labels=item['label'][None];e=FIRST[0]
        z=torch.zeros(1,3,*labels.shape[2:],requires_grad=True)
        t=target(item,e);v=components(z,labels,e,t)
        g=torch.autograd.grad(v['ap_band']+v['ap_pair'],z)[0]
        self.assertTrue((g[:,0]==0).all())
        self.assertTrue((g[0,:,labels[0,0]==0]==0).all())
        self.assertTrue((g[0,1,labels[0,0]==1]<=0).all())
        self.assertTrue((g[0,1,labels[0,0]==2]>=0).all())
        self.assertGreater(float(g.abs().sum()),0.)
        # Changing the other endpoint must not alter this endpoint's gradient.
        t=deepcopy(t);t['ap82']['pairs']=[dict(inside=dict(index=torch.tensor([0]),weight=torch.ones(1)),
            outside=dict(index=torch.tensor([1]),weight=torch.ones(1))) for _ in range(2)]
        endpoint=deepcopy(e);endpoint['ap82']['band_share']=0.
        a=components(z,labels,endpoint,t)['ap_pair'];ga=torch.autograd.grad(a,z)[0]
        zz=z.detach().clone();zz[0,1].reshape(-1)[1]=3.;zz.requires_grad_()
        gb=torch.autograd.grad(components(zz,labels,endpoint,t)['ap_pair'],zz)[0]
        torch.testing.assert_close(ga[0,:,0,0,0],gb[0,:,0,0,0],atol=0,rtol=0)

    def test_all_74_values_gradients_budgets_and_frozen_backgrounds(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        settings=dict(prefix_epochs=5,target_ratio=.1,cap_ratio=.5)
        for e in FIRST+SECOND:
            t=target(item,e);values=components(z,labels,e,t)
            reference=base_components(z,labels,e['base_objective'],t['base'])
            for name,value in reference.items():
                torch.testing.assert_close(values[name],value,atol=0,rtol=0)
                torch.testing.assert_close(torch.autograd.grad(values[name],z,retain_graph=True)[0],
                    torch.autograd.grad(value,z,retain_graph=True)[0],atol=0,rtol=0)
            result=select_weights([gradient_record(values,z)],e,settings)
            weights=result['coefficients']
            for name,value in e['base_coefficients'].items():self.assertEqual(weights[name],value)
            base=combined_loss(reference,e['base_coefficients'])
            added=sum(weights[k]*v for k,v in values.items() if k.startswith('ap_'))
            gb=torch.autograd.grad(base,z,retain_graph=True)[0]
            ga=torch.autograd.grad(added,z,retain_graph=True)[0]
            self.assertAlmostEqual(float(ga.norm()/gb.norm()),e['ap82']['gradient_ratio'],places=5)
            self.assertTrue(torch.isfinite(ga).all())

    def test_absent_classes_and_perfect_conditional_predictions(self):
        e=FIRST[1];shape=(9,9,9)
        for value in (0,1,2):
            item=dict(case_name='empty',label=torch.full((1,*shape),value),affine=np.eye(4))
            z=torch.randn(1,3,*shape,requires_grad=True)
            values=components(z,item['label'][None],e,target(item,e))
            loss=values['ap_band']+values['ap_pair']
            self.assertEqual(float(loss),0.);self.assertEqual(float(torch.autograd.grad(loss,z)[0].abs().sum()),0.)
        item=synthetic_item();labels=item['label'][None]
        z=torch.zeros(1,3,*labels.shape[2:]);z[:,0]=30 # Conditional AP supervision is not FG supervision.
        z[:,1]=torch.where(labels[:,0]==1,15.,-15.);z[:,2]=-z[:,1];z.requires_grad_()
        values=components(z,labels,e,target(item,e))
        self.assertLess(float(values['ap_band']+values['ap_pair']),1e-20)

    def test_exact_restart_with_shared_old_history_and_both_backgrounds(self):
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=5,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        with tempfile.TemporaryDirectory() as directory,patch.object(train,'_load_pkl_dataframe',return_value=None),\
             patch.object(train,'Patients',TinyPatients),patch.object(train,'model_factory',side_effect=tiny_model):
            root=Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):train.run(p,root/'prefix',BASES['dice'],stop_after=2,device_override='cpu')
            with installed():
                for e in FIRST[:2]:
                    for mode in ('whole','resume'):
                        for file in ('latest.pt','best.pt'):
                            state=torch.load(root/'prefix/runs/dice'/file,weights_only=False)
                            state['identity']=train.identity(p,e)
                            folder=root/mode/'runs'/e['id'];folder.mkdir(parents=True,exist_ok=True)
                            save_checkpoint(folder/file,state)
                    with contextlib.redirect_stdout(io.StringIO()):
                        a=train.run(p,root/'whole',e,device_override='cpu')
                        train.run(p,root/'resume',e,stop_after=3,device_override='cpu')
                        b=train.run(p,root/'resume',e,device_override='cpu')
                    self.assertEqual(a['validation'],b['validation'])
                    self.assertEqual(a['best_epoch'],b['best_epoch'])
                    aa=torch.load(root/'whole/runs'/e['id']/'best.pt',weights_only=False)
                    bb=torch.load(root/'resume/runs'/e['id']/'best.pt',weights_only=False)
                    for k in aa['model']:torch.testing.assert_close(aa['model'][k],bb['model'][k],atol=0,rtol=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(APTests))
    if not result.wasSuccessful():raise SystemExit(1)
    value=dict(passed=True,tests=result.testsRun,stage1_count=30,stage2_count=44,stage3_max=32,
        source_hashes={n:hashlib.sha256((ROOT/'scripts/band_focus'/n).read_bytes()).hexdigest() for n in SOURCES})
    args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)


if __name__=='__main__':main()
