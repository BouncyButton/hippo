import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from scripts.colab_hfh.nnunet_search13_policy import ARMS,CAP,ACTIVITY,candidates,make_protocol,active_coefficients,probe_allowance
from scripts.colab_hfh.constraint_calibration import components
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.nnunet_search13_worker import source

ROOT = Path(__file__).resolve().parents[2]


class PolicyTests(unittest.TestCase):
    def test_search_complete_bounded_and_no_absent_emphasis(self):
        rows = candidates()
        self.assertEqual(len(rows),78)
        self.assertEqual(len({r['id'] for r in rows}),78)
        self.assertEqual({r['arm'] for r in rows},set(ARMS))
        for row in rows:
            self.assertTrue(row['emphasis'] is None or row['emphasis'] in components(row['arm']))

    def test_activity_floor_under_drift_and_cancellation(self):
        random = np.random.default_rng(199)
        for count in range(1,5):
            names = ['bands','edge','sdf','depth'][:count]
            for _ in range(100):
                a = random.normal(size=(count+1,count+1))
                gram = a@a.T + np.eye(count+1)*.01
                weights = {n:float(10**random.uniform(-2,0)) for n in names}
                w,stats = active_coefficients(gram,weights,names)
                self.assertGreaterEqual(min(stats['component_ratios'].values()),.01-1e-10)
                self.assertGreaterEqual(min(stats['component_shares'].values()),.10-1e-10)
                self.assertLessEqual(stats['sum_ratio'],.5+1e-10)
                self.assertLessEqual(stats['joint_ratio'],.5+1e-10)
                self.assertTrue(all(v>0 for v in w.values()))

    def test_feasible_weights_unchanged_and_invalid_fail_closed(self):
        w,stats = active_coefficients(np.eye(3),{'bands':.05,'edge':.08},['bands','edge'])
        self.assertEqual(w,{'bands':.05,'edge':.08})
        self.assertFalse(stats['adjusted'])
        with self.assertRaises(ValueError):active_coefficients(np.diag([1.,0.]),{'sdf':.1},['sdf'])
        with self.assertRaises(ValueError):active_coefficients(np.diag([1.,1e-20]),{'sdf':.1},['sdf'])
        with self.assertRaises(ValueError):active_coefficients([[1,3],[3,1]],{'sdf':.1},['sdf'])

    def test_protocol_same_cap_with_probe_charged_and_same_native_prefix(self):
        for recipe in candidates():
            arm = recipe['arm']
            ref = dict(active_objective=arm,budget_flops=CAP,models={'nnunet':{'objectives':{arm:dict(
                prefix_updates=60,sdf_prefix_updates=60,baseline_flops_per_update=355237172535,
                early_flops_per_update=355400000000,flops_per_update=355400000000,calibration_flops=1200000000000)}}})
            p = make_protocol(ref,recipe,'/anchor.json')
            s = p['models']['nnunet']['objectives'][arm]
            self.assertEqual(p['folds'],[0]);self.assertEqual(p['seeds'],[0])
            self.assertEqual(training_flops(60,s),60*355237172535)
            self.assertLessEqual(training_flops(s['max_steps'],s),CAP)
            self.assertGreater(training_flops(s['max_steps']+1,s),CAP)
            self.assertEqual(s['flops_per_update'],355400000000+probe_allowance(arm))
            self.assertGreaterEqual(s['max_steps']-60-recipe['ramp'],60)

    def test_generated_workers_compile(self):
        for native in (True,False):
            text = source(native)
            compile(text,'worker','exec')
            self.assertIn('1, True',text)
            if not native:self.assertIn('ActiveLoss(exp.trainer.training_loss',text)


class RuntimeTests(unittest.TestCase):
    def test_actual_installer_and_native_hooks(self):
        from scripts.colab_hfh.nnunet_search13_worker import install,install_native
        from scripts.colab_hfh.nnunet_seed0_long_worker import SCHEDULE
        from scripts.colab_hfh.nnunet_warm_ready import READY_POLICY
        path = ROOT/'scripts/cross_dataset_campaign/capped_core.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for native in (True,False):
            ns = {};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
            trainer = SimpleNamespace(run=ns['run'],calibrate=lambda *a,**k:None,
                                      base=SimpleNamespace(),__file__=str(path))
            p = dict(long_schedule=SCHEDULE,augmentation='none',lr_horizon_updates=180,
                     long_source_sha256={},warm_ready_source_sha256={},readiness_policy=READY_POLICY,
                     constraint_component_targets={'bands':.1},calibration_p95_cap=.5,
                     constraint_joint_target=.1,constraint_joint_cap=.5)
            (install_native if native else install)(trainer,p,'dice' if native else 'dice_bands')
            self.assertIn('adherence',trainer.run.__code__.co_names)
            self.assertEqual(trainer.base.learning_rate('nnunet',100,180,.1),.07)

    def test_real_loss_probe_and_other_native_heads_preserved(self):
        import torch
        from functools import partial
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from scripts.adni_constraints import objectives as binary
        from scripts.nnunet_band_objective import native_loss
        from scripts.colab_hfh.nnunet_search13_worker import measure,ActiveLoss,terms
        from scripts.colab_hfh.constraint_geometry import sdf_targets,depth_weights,DEPTH
        torch.set_num_threads(1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = DeepSupervisionWrapper(DC_and_CE_loss(
            {'batch_dice':True,'smooth':1e-5,'do_bg':False,'ddp':False},{},dice_class=MemoryEfficientSoftDiceLoss),[4/7,2/7,1/7,0])
        torch.manual_seed(11)
        outputs = [torch.randn(1,2,n,n,n,device=device,requires_grad=True) for n in (64,32,16,8)]
        labels = torch.zeros(1,1,64,64,64,device=device,dtype=torch.long)
        labels[:,:,20:40,25:35,20:40] = 1
        sdf = sdf_targets(labels)
        depth = depth_weights(labels,np.diag([1.,2.,1.,1.]),DEPTH)
        arm = 'dice_bands_edge_sdf_depth'
        gram,_ = measure(outputs,labels,arm,sdf,criterion,depth)
        ratios = np.sqrt(np.diag(gram)[1:]/gram[0][0])
        anchor = {n:.04/float(r) for n,r in zip(components(arm),ratios)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'budget_ledger.json').write_text(json.dumps({'charged_attempts':70}))
            original = partial(binary.training_loss,policy='original_shared_ramp')
            wrapper = ActiveLoss(original,dict(active_objective=arm,activity_policy=ACTIVITY),output)
            value = wrapper(outputs,labels,arm,anchor,1.,sdf,criterion,depth,1.)
            row = json.loads((output/'constraint_activity.jsonl').read_text())
            expected = native_loss(outputs,labels,criterion)+sum(row['effective_weights'][n]*v for n,v in terms(outputs[0],labels,arm,sdf,depth).items())
            self.assertTrue(torch.allclose(value,expected,rtol=1e-6,atol=1e-7))
            actual = torch.autograd.grad(value,outputs,retain_graph=True,allow_unused=True)
            native = torch.autograd.grad(native_loss(outputs,labels,criterion),outputs,retain_graph=True,allow_unused=True)
            for a,b in zip(actual[1:],native[1:]):
                self.assertTrue(a is None and b is None or torch.equal(a,b))
            self.assertTrue(all(v>=.01-1e-9 for v in row['component_ratios'].values()))
            untouched = wrapper(outputs,labels,'dice',{},.2,None,criterion,None,0.)
            self.assertTrue(torch.equal(untouched,native_loss(outputs,labels,criterion)))


if __name__ == '__main__':
    runtime = '--runtime' in sys.argv
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests)
    if runtime:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
