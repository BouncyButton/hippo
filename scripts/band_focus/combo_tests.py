"""Loss equivalence, joint gradients, target edge cases and exact checkpoint resume."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import combo_matrix as matrix, combo_train as train
from .combo_objectives import Targets, components, select_weights
from .winner_objectives import components as base_components
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.native_branch_constraints.objectives import gradient_record
from scripts.depth_supervision.losses import combined_loss, components as depth_components
from scripts.depth_supervision.geometry import build_geometry
from scripts.depth_supervision.utils import seed_all, save_checkpoint
from thesis.new_constraints.edge_consistency import BoundaryEdgeConsistencyLoss
from thesis.new_constraints.signed_distance import signed_distance_loss, signed_distance_maps


def example_base():
    # Keep this fixture independent of validation data and local saved protocols.
    return dict(id=matrix.ANCHOR,family='w81_strength',objective_version='winner_v1',
        band=dict(kind='regional',inner=3,outer=2,penalty='brier',inside_weight=.5,
                  inside_fraction=.7,outside_fraction=.1,inside_hard_mix=.65,outside_hard_mix=.85),
        pair=dict(distances=[1,3],weights=[.5,.5],support='all',form='independent'),
        surface=None,ap=None,ap_ce=None,band_budget=1.25,extra_budget=.75,
        extra_shares=None,focus_schedule=None,training=None,compare='unused',
        fraction='5pct',seed=0,arm='ordinary_swinunetr')


BASE = example_base()
ENTRIES = matrix.entries(BASE,dict(bands=.04,pair=.006))


class ComboTests(unittest.TestCase):
    def setUp(self):seed_all(0)

    def test_matrix_frozen_base_and_complete_factorial(self):
        self.assertEqual(len(ENTRIES),22)
        self.assertEqual(ENTRIES[0]['combo']['terms'],[])
        for ratio in (.025,.05,.1):
            seen={tuple(e['combo']['terms']) for e in ENTRIES if e['combo']['gradient_ratio']==ratio}
            self.assertEqual(seen,set(matrix.SUBSETS))
        for e in ENTRIES:
            self.assertEqual(e['base_objective'],BASE)
            self.assertEqual(e['base_coefficients'],dict(bands=.04,pair=.006))
            self.assertFalse(e.get('training'))
        self.assertEqual(ENTRIES,json.loads(json.dumps(ENTRIES)))

    def test_original_terms_and_exact_base_gradients_all_combinations(self):
        item=synthetic_item();labels=item['label'][None]
        p=dict(native_geometry={item['case_name']:dict(native_shape=list(labels.shape[2:]))},geometry={})
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        with tempfile.TemporaryDirectory() as d:
            t=Targets(Path(d),p,ENTRIES[-1]).get(item,'cpu')
            reference=base_components(z,labels,BASE,t['base'])
            extras=dict(sdf=signed_distance_loss(z,labels,t['extra']['sdf']),
                        edge=BoundaryEdgeConsistencyLoss()(z,labels),
                        depth=depth_components(z,labels,t['extra'],'depth')['depth'])
            for e in ENTRIES:
                values=components(z,labels,e,t)
                self.assertEqual(set(values),set(reference)|set(e['combo']['terms']))
                for name,value in values.items():
                    expected=reference[name] if name in reference else extras[name]
                    torch.testing.assert_close(value,expected,rtol=0,atol=0)
                    torch.testing.assert_close(torch.autograd.grad(value,z,retain_graph=True)[0],
                        torch.autograd.grad(expected,z,retain_graph=True)[0],rtol=0,atol=0)
                self.assertEqual(set(components(z,labels,e,{},active=False)),{'dice'})
            # No gradients may flow into fixed annotation distance targets.
            self.assertFalse(t['extra']['sdf'].requires_grad)
            self.assertAlmostEqual(float(t['extra']['weights'].sum()),1.,places=6)

    def test_joint_strength_not_multiplied_by_number_of_terms(self):
        item=synthetic_item();labels=item['label'][None]
        p=dict(native_geometry={item['case_name']:dict(native_shape=list(labels.shape[2:]))},geometry={})
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        with tempfile.TemporaryDirectory() as d:
            t=Targets(Path(d),p,ENTRIES[-1]).get(item,'cpu')
            full=components(z,labels,ENTRIES[-1],t);rows=[gradient_record(full,z)]
            settings=dict(prefix_epochs=5,cap_ratio=.5)
            for e in ENTRIES:
                weights=select_weights(rows,e,settings)['coefficients']
                for k,v in e['base_coefficients'].items():self.assertEqual(weights[k],v)
                base=combined_loss({k:full[k] for k in ('dice','bands','pair')},e['base_coefficients'])
                if not e['combo']['terms']:
                    actual=combined_loss(components(z,labels,e,t),weights)
                    torch.testing.assert_close(actual,base,rtol=0,atol=0)
                    continue
                extra=sum(weights[k]*full[k] for k in e['combo']['terms'])
                a=torch.autograd.grad(extra,z,retain_graph=True)[0]
                b=torch.autograd.grad(base,z,retain_graph=True)[0]
                self.assertAlmostEqual(float(a.norm()/b.norm()),e['combo']['gradient_ratio'],places=5)
                self.assertTrue(torch.isfinite(a).all())

    def test_missing_classes_sdf_and_depth_sparse_bins(self):
        for value in (0,1,2):
            y=torch.full((1,1,9,9,9),value)
            phi=signed_distance_maps(y)
            z=torch.zeros((1,3,9,9,9),requires_grad=True)
            loss=signed_distance_loss(z,y,phi)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss,z)[0]).all())
        item=synthetic_item();g=build_geometry(item['label'][0].numpy(),item['affine'],fields=False)
        self.assertAlmostEqual(float(g['weights'].sum()),1.,places=6)
        actual=np.array([g['weights'][g['bin_ids']==b].sum() for b in g['bin_values']])
        np.testing.assert_allclose(actual,g['bin_masses'],rtol=1e-5,atol=1e-7)
        y=item['label'][None];phi=signed_distance_maps(y)
        onehot=torch.nn.functional.one_hot(y[:,0].long(),3).permute(0,4,1,2,3).float()
        z=(40*onehot-20).requires_grad_()
        self.assertLess(abs(float(signed_distance_loss(z,y,phi))),1e-12)
        self.assertLess(float(BoundaryEdgeConsistencyLoss()(z,y)),1e-12)

    def test_training_loop_only_intended_edits(self):
        path=Path(__file__).parent
        old=(path/'train.py').read_text()
        expected=old.replace('"""Train matched band/Chamfer follow-ups from a common Dice-only warmup."""',
            '"""Frozen training loop with established auxiliary additions and truthful metadata."""')
        expected=expected.replace('from .geometry import Targets','from .combo_objectives import Targets, write_history')
        expected=expected.replace('from .objectives import components as objective_components, at_epoch',
            'from .combo_objectives import components as objective_components\nfrom .objectives import at_epoch')
        expected=expected.replace('from .objectives import select_weights','from .combo_objectives import select_weights')
        expected=expected.replace('TERM_ORDER = ("bands", "pair", "surface", "ap", "ap_ce")',
            'TERM_ORDER = ("bands", "pair", "sdf", "edge", "depth")')
        expected=expected.replace('objective=objective, depth_balancing=False,',
            "objective=objective, depth_balancing='depth' in entry['combo']['terms'],")
        self.assertEqual((path/'combo_train.py').read_text(),expected)

    def test_exact_restart_and_old_warmup_history(self):
        from . import train as old_train
        dice=dict(BASE,id='dice',objective_version='original',band=dict(kind='none'),pair=None)
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=5,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        with tempfile.TemporaryDirectory() as directory,contextlib.ExitStack() as stack:
            for module in (train,old_train):
                stack.enter_context(patch.object(module,'_load_pkl_dataframe',return_value=None))
                stack.enter_context(patch.object(module,'Patients',TinyPatients))
                stack.enter_context(patch.object(module,'model_factory',side_effect=tiny_model))
            root=Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):old_train.run(p,root/'prefix',dice,stop_after=2,device_override='cpu')
            for e in (ENTRIES[0],ENTRIES[-1]):
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
                self.assertEqual(a['validation'],b['validation']);self.assertEqual(a['best_epoch'],b['best_epoch'])
                aa=torch.load(root/'whole/runs'/e['id']/'best.pt',weights_only=False)
                bb=torch.load(root/'resume/runs'/e['id']/'best.pt',weights_only=False)
                for k in aa['model']:torch.testing.assert_close(aa['model'][k],bb['model'][k],atol=0,rtol=0)
                config=json.loads((root/'whole/runs'/e['id']/'config.json').read_text())
                self.assertEqual(config['depth_balancing'],'depth' in e['combo']['terms'])

    def test_replication_selection_and_fold_isolation(self):
        from .combo_replication import select_candidate,validate_inputs
        candidates=[dict(objective='a',macro_dice=.813,max_epochs=75,step_size=20),
                    dict(objective='b',macro_dice=.814,max_epochs=75,step_size=20),
                    dict(objective='extended',macro_dice=.9,max_epochs=150,step_size=40)]
        self.assertEqual(select_candidate(candidates)['objective'],'b')
        inputs=dict(pkl_sha256='hash',splits={str(f):dict(train=[f't{i}' for i in range(10)],
            val=[f'v{f}_{i}' for i in range(52)]) for f in range(3)})
        inputs['native_geometry']={n:{} for s in inputs['splits'].values() for n in s['train']+s['val']}
        p=dict(pkl_sha256='hash',splits={'5pct':inputs['splits']['0']},native_geometry={})
        validate_inputs(inputs,p)
        changed=json.loads(json.dumps(inputs));changed['splits']['1']['train'][0]='v1_0'
        with self.assertRaises(AssertionError):validate_inputs(changed,p)

    def test_replication_recalibrates_base_from_own_training_gradients(self):
        from . import combo_replication as replication,objectives
        item=synthetic_item();labels=item['label'][None]
        p=dict(native_geometry={item['case_name']:dict(native_shape=list(labels.shape[2:]))},geometry={})
        settings=dict(prefix_epochs=5,target_ratio=.1,cap_ratio=.5)
        ap=dict(BASE,objective_version='ap_boundary_v1',base_objective=BASE,
                base_coefficients=dict(bands=.04,pair=.006),combo=dict(terms=[]),
                ap82=dict(width=3.,anterior_fraction=.5,posterior_fraction=.5,
                    anterior_emphasis=.5,posterior_emphasis=.5,anterior_mass=.5,
                    endpoint_weights=[.5,.5],band_share=.25,gradient_ratio=.025))
        with tempfile.TemporaryDirectory() as d:
            for e in (BASE,ap,ENTRIES[-1]):
                z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
                t=replication.targets(Path(d),p,e).get(item,'cpu')
                values=replication.components(z,labels,e,t)
                rows=[gradient_record(values,z)]
                result=replication.select_weights(rows,e,settings)
                expected=objectives.select_weights(rows,e.get('base_objective',e),settings)
                for k,v in expected['coefficients'].items():self.assertEqual(result['coefficients'][k],v)
                self.assertTrue(all(np.isfinite(v) and v>0 for v in result['coefficients'].values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComboTests))
    path=Path(__file__).parent
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        source_hashes={n:hashlib.sha256((path/n).read_bytes()).hexdigest() for n in matrix.SOURCES}),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
