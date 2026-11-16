"""Readiness, fixed compute, matching, and trainer integration checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.colab_hfh.nnunet_warm_ready import lr,readiness,payload


class PolicyTests(unittest.TestCase):
    def test_warmup_and_decay(self):
        self.assertEqual([round(lr(i),6) for i in range(5)],[.006,.012,.018,.024,.03])
        self.assertEqual(lr(5),.03)
        self.assertTrue(all(lr(i)>lr(i+1)>0 for i in range(5,89)))

    def test_panel_not_single_case(self):
        rows=[dict(case_name=str(i),hard_dice=.6) for i in range(10)]
        self.assertTrue(readiness(rows))
        self.assertFalse(readiness(rows[:9]))
        self.assertFalse(readiness([rows[0]]*10))
        for r in rows[:3]:r['hard_dice']=.4
        self.assertFalse(readiness(rows))

    def test_freeze_budget(self):
        root=Path('reports/hfh_colab_exploration_20261111')
        base=json.loads((root/'bands_v1_payload.json').read_text())
        plan=json.loads((root/'constraint_matrix_plan.json').read_text())
        before=copy.deepcopy(base)
        for arm,n in [('dice',90),('dice_bands',89)]:
            result=payload(base,plan,arm);p=result['protocols']['nnunet']['protocol.json']
            s=p['models']['nnunet']['objectives'][arm]
            self.assertEqual(s['max_steps'],n)
            self.assertLessEqual(n*s['flops_per_update'],31976450667090)
            self.assertGreater((n+1)*s['flops_per_update'],31976450667090)
            self.assertEqual(p['initial_lrs']['nnunet'],.03)
            self.assertEqual(p['nnunet_warmup_fraction'],5/90)
            self.assertNotIn('min_pair_delta_pp',result)
        self.assertEqual(base,before)

    def test_worker_and_actual_trainer_integration(self):
        from scripts.colab_hfh.nnunet_warm_ready_worker import worker_source,install
        import types
        compile(worker_source(),'generated_worker','exec')
        root=Path('reports/hfh_colab_exploration_20261111')
        base=json.loads((root/'bands_v1_payload.json').read_text());plan=json.loads((root/'constraint_matrix_plan.json').read_text())
        # Compile the actual run function without importing GPU-only dependencies.
        import ast
        path=Path('scripts/cross_dataset_campaign/capped_core.py')
        source=path.read_text();tree=ast.parse(source)
        node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        import linecache
        filename='/tmp/hfh_ready_test_trainer.py';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),ns)
        for arm in ('dice','dice_bands'):
            trainer=types.ModuleType('test_trainer');trainer.__file__=filename
            trainer.run=ns['run'];trainer.calibrate=lambda *a,**kw:{}
            install(trainer,payload(base,plan,arm)['protocols']['nnunet']['protocol.json'],arm)
            self.assertTrue(callable(trainer.run))

    def test_readiness_monitor_preserves_native_update(self):
        import torch
        from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss
        from scripts.colab_hfh.bands_campaign import write
        torch.manual_seed(4)
        model=torch.nn.Conv3d(1,2,1);reference=copy.deepcopy(model)
        x=torch.rand(1,1,8,32,8);y=torch.zeros(1,1,8,32,8,dtype=torch.long);y[:,:,:,17:22,:]=1
        original=lambda outputs,labels,*a,**kw:torch.nn.functional.cross_entropy(outputs,labels[:,0])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);write(path/'budget_ledger.json',dict(charged_attempts=1))
            wrapped=ReadyLoss(original,path,{})
            wrapped.bind_cases([dict(label=y[0],case_name='training')]);wrapped.model=model
            a=torch.optim.SGD(model.parameters(),lr=.006,momentum=.95)
            b=torch.optim.SGD(reference.parameters(),lr=.006,momentum=.95)
            original(reference(x),y).backward();b.step()
            wrapped(model(x),y,'dice',{},0).backward();a.step();wrapped.finish(x,y,'cpu',False)
            for p,q in zip(model.parameters(),reference.parameters()):self.assertTrue(torch.equal(p,q))
            resumed=ReadyLoss(original,path,{})
            self.assertFalse(resumed.ready)
            with self.assertRaises(ValueError):resumed(model(x),y,'dice',{},0)


if __name__=='__main__':unittest.main()
