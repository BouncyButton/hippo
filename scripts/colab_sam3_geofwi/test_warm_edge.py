"""CPU regression checks for adapted calibration, exact budget and recovery."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.colab_sam3_geofwi import warm_edge as w, diagnose as d, runtime as rt
from scripts.colab_sam3_geofwi.test_diagnose import Tiny, setup_toy, digest, reset


class WarmEdgeTests(unittest.TestCase):
    def test_weight_rule_keeps_bands_fixed_and_caps_p95(self):
        rows=[dict(gram=np.diag([4.,9.,v*v]).tolist()) for v in [1.,1.,1.,1.,100.]]
        weight,stats=w.choose_edge_weight(rows,2.)
        raw=np.array([1,1,1,1,100])/np.sqrt(40)
        self.assertAlmostEqual(weight,min(.1/np.median(raw),.5/np.quantile(raw,.95)))
        self.assertLessEqual(stats['weighted_p95'],.5+1e-12)

    def run_fixture(self,base,stop_kind=None):
        output=base/'work';recovery=base/'recovery';durable=base/'durable'
        output.mkdir(parents=True)
        original,slabs=setup_toy()
        p=dict(updates=25,accumulation=2,seed=0,calibration_seed=4,train_cases=list(slabs.indices),
               warmup_updates=20,constraint_warmup_updates=20,learning_rate=.01,final_learning_rate=.001)
        rt.write(output/'protocol.json',p)
        shared=dict(rows=[dict(case=c,anchor=a) for c in p['train_cases'] for a in (0,1,2,2)])
        enabled=[stop_kind is not None]
        def stopping():
            if not enabled[0]:return False
            h=rt.read(output/'training_history.json') if (output/'training_history.json').exists() else []
            if isinstance(stop_kind,int):return len(h)>=stop_kind
            path=durable/'probes/calibration_update5.json'
            return path.exists() and len(rt.read(path)['rows'])>=7
        original.stopping=stopping
        def run():
            model=Tiny();writer=rt.RecoveryWriter(output,recovery,interval=99999)
            try:w.train(original,d,rt,model,None,[model.lora_A],slabs,p,shared,2.,output,recovery,writer,durable)
            finally:writer.close()
            return model
        if stop_kind is not None:
            with self.assertRaises(SystemExit) as caught:run()
            self.assertEqual(caught.exception.code,75)
            # Restore from actual Drive-style recovery tar, including optimizer/RNG/calibration.
            import shutil
            shutil.rmtree(output);rt.restore(output,recovery)
            enabled[0]=False
        final=run()
        history=rt.read(output/'training_history.json');cal=rt.read(output/'calibration.json')
        self.assertEqual(len(history),25)
        self.assertTrue(all(x['edge_weight']==0 for x in history[:5]))
        self.assertTrue(all(x['edge_weight']==cal['weights']['edge']>0 for x in history[5:]))
        self.assertTrue(all(x['bands_weight']==2. for x in history))
        self.assertEqual(len(cal['rows']),20)
        self.assertEqual({k:len(v) for k,v in cal['monitoring'].items()},{'10':5,'25':5})
        # A rerun for interrupted evaluation must not change the checkpoint identity.
        checkpoint_sha=rt.digest(output/'checkpoint_latest.pt')
        rerun=run();self.assertEqual(rt.digest(output/'checkpoint_latest.pt'),checkpoint_sha)
        self.assertEqual(digest(final,True),digest(rerun,True))
        # Compare the five-update prefix to a direct bands-only loop with the original ramp.
        baseline=Tiny();reset(0);optimizer=torch.optim.AdamW([baseline.lora_A],lr=.01,weight_decay=.01)
        schedule=slabs.schedule(p)
        for i in range(5):
            optimizer.param_groups[0]['lr']=original.lr_for_update(i,200,20,.01,.001)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                case,anchor=schedule[i*2+micro]
                losses=original.forward_slab(baseline,None,slabs,case,anchor)
                ((losses['native']+(i+1)/20*(2*losses['bands']+0*losses['edge']))/2).backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.);optimizer.step()
        self.assertEqual(cal['adapter_at_calibration'],digest(baseline,True))
        return history,digest(final,True),cal

    def test_full_run_resume_during_prefix_calibration_and_edge_training(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            base=Path(tmp)
            expected=self.run_fixture(base/'full')
            for stop in (3,'calibration',12):
                actual=self.run_fixture(base/str(stop),stop)
                self.assertEqual(actual,expected)

    def test_notebook_only_launches_one_retry(self):
        import ast
        from scripts.colab_sam3_geofwi.prepare_warm_edge import EXPORT
        nb=rt.read(EXPORT/'SAM3_GeoFWI3D_warm_edge.ipynb')
        sources='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        self.assertNotIn("campaign('preflight')",sources)
        self.assertNotIn("BUNDLE/'parallel.py'",sources)
        self.assertNotIn('DIAGNOSTIC_SCRIPT',sources)
        self.assertIn('results_5vol_25updates_warm5_edge10_v1',sources)
        self.assertEqual(sources.count('execute([PYTHON,RETRY_SCRIPT'),1)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])

    def test_main_evaluates_only_new_model_archives_and_skips_completed(self):
        from scripts.colab_sam3_geofwi import parallel
        from scripts.cross_dataset_campaign import common
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)/'bundle';prior=Path(tmp)/'prior';durable=Path(tmp)/'results'
            root.mkdir();rt.write(root/'bundle_manifest.json',{})
            rt.write(prior/'environment_binding.json',{})
            original,slabs=setup_toy();original.stopping=lambda:False
            tasks=[dict(id='bands',scientific_arm='dice_bands',updates=25),
                   dict(id='edge',scientific_arm='dice_bands_edge',updates=25)]
            rt.write(root/'plan.json',dict(tasks=tasks))
            p=dict(updates=25,accumulation=2,seed=0,calibration_seed=4,train_cases=list(slabs.indices),
                evaluation_cases=[f'val{i}' for i in range(96)],warmup_updates=20,
                constraint_warmup_updates=20,learning_rate=.01,final_learning_rate=.001)
            shared=dict(rows=[dict(case=c,anchor=a) for c in p['train_cases'] for a in (0,1,2,2)],
                inventory=slabs.inventory,initialization=dict(frozen=digest(Tiny(),False),adapter=digest(Tiny(),True)))
            rt.write(prior/'calibration/fold0/calibration.json',shared)
            records=[dict(case_name=c,split=s,macro_hard_dice=.7,truth_sha256=c,
                          confusion=[[1,1],[1,1]],foreground_fn=1,foreground_fp=1)
                     for s,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])] for c in cases]
            done=dict(validation_dice=.7,training_dice=.7,foreground_fn=96,foreground_fp=96)
            for task in tasks:
                folder=prior/'runs'/task['id']
                rt.write(folder/'completion.json',done);rt.write(folder/'evaluation.json',dict(records=records))
                rt.write(folder/'calibration.json',dict(weights=dict(bands=2.,edge=10.)))
            def load(*_):
                model=Tiny();return original,model,None,[model.lora_A],None,None,slabs
            evaluations=[]
            def evaluate(original,model,processor,volumes,p,output,root,durable):
                evaluations.append(1)
                result=dict(records=records,mean_dice=dict(training_support=.7,validation=.7))
                rt.write(output/'evaluation.json',result);return result
            engine=SimpleNamespace(load=load,evaluate=evaluate)
            real_completed=rt.completed
            def completed(root,where,task):return True if where==prior else real_completed(root,where,task)
            def binding(root,where):rt.write(where/'environment_binding.json',{});return {}
            argv=['retry','--bundle',str(root),'--source-results',str(prior),
                  '--previous-results',str(Path(tmp)/'baseline'),'--durable',str(durable)]
            with patch.dict(sys.modules,{'runtime':rt,'engine':engine,'parallel':parallel,'sam3_edge_diagnose_v1':d}),\
                 patch.object(rt,'activate'),patch.object(rt,'verify_bundle',return_value=dict(tasks=tasks)),\
                 patch.object(rt,'binding',side_effect=binding),patch.object(rt,'fold_calibration',return_value=shared),\
                 patch.object(rt,'previous_baseline',return_value=(done,records)),patch.object(rt,'quota'),\
                 patch.object(rt,'protocol',side_effect=lambda *_:dict(p)),patch.object(rt,'completed',side_effect=completed),\
                 patch.object(common,'install_signals'),patch.object(sys,'argv',argv):
                w.main();w.main()
            self.assertEqual(len(evaluations),1)
            self.assertEqual(rt.read(durable/'comparison.json')['new_runs'],1)
            self.assertEqual(len(rt.read(durable/'comparison.json')['rows']),4)
            self.assertFalse((root/'work/edge_warm5_edge_v1/checkpoint_latest.pt').exists())


if __name__=='__main__':unittest.main()
