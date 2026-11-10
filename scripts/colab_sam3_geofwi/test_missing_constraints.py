"""CPU checks for matrix scope, selected losses, adapted calibration and recovery."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

from scripts.colab_sam3_geofwi import missing_constraints as m, diagnose as d, runtime as rt
from scripts.colab_sam3_geofwi.test_diagnose import Tiny,setup_toy,digest
from scripts.cross_dataset_campaign.common import ARMS


def fixture(arm):
    original,slabs=setup_toy();old=original.forward_slab
    def forward(model,criterion,slabs,case,anchor):
        losses=old(model,criterion,slabs,case,anchor)
        # Distinct, nonzero differentiable fields exercise routing and negative SDF loss.
        losses['sdf']=model.lora_A[0]-model.lora_A[2]
        losses['depth']=(model.lora_A[1]+.4).square()
        return losses
    original.forward_slab=forward
    p=dict(arm=arm,updates=25,accumulation=2,seed=0,calibration_seed=4,
           train_cases=list(slabs.indices),evaluation_cases=[f'val{i}' for i in range(96)],
           warmup_updates=20,constraint_warmup_updates=20,learning_rate=.01,final_learning_rate=.001)
    slabs.protocol=p;slabs.sdf={c:torch.ones(3,7,9) for c in p['train_cases']}
    slabs.depth={c:torch.ones(3,7,9) for c in p['train_cases']}
    shared=dict(rows=[dict(case=c,anchor=a) for c in p['train_cases'] for a in (0,1,2,2)],
        inventory=slabs.inventory,initialization=dict(frozen=digest(Tiny(),False),adapter=digest(Tiny(),True)))
    return original,slabs,p,shared


class MissingTests(unittest.TestCase):
    def test_exact_missing_matrix(self):
        self.assertEqual(m.ARMS,tuple(a for a in ARMS if a not in ('dice','dice_bands','dice_bands_edge')))
        self.assertEqual(len(m.tasks_for(dict(id='old',updates=25))),11)
        for arm in ('dice','dice_bands','dice_bands_edge','dice_edge'):
            with self.assertRaises(ValueError):m.names_for(arm)

    def test_sequential_calibration_and_combined_ratio(self):
        names=['native','bands','edge','sdf','depth']
        rows=[dict(gram=np.diag([4.,9.,16.,25.,36.]).tolist()) for _ in range(20)]
        weights,stats=m.choose_weights(rows,names,2.)
        self.assertEqual(weights['bands'],2.)
        previous=40.
        for n,v in [('edge',16.),('sdf',25.),('depth',36.)]:
            expected=.1*np.sqrt(previous/v)
            self.assertAlmostEqual(weights[n],expected)
            self.assertAlmostEqual(stats[n]['weighted_median'],.1)
            previous+=expected**2*v
        actual=m.ratio_metrics(rows[0]['gram'],names,weights)
        self.assertAlmostEqual(actual['combined_auxiliary_over_native'],np.sqrt((previous-4)/4))

    def train_fixture(self,base,arm,stop=None):
        original,slabs,p,shared=fixture(arm)
        output=base/'run';recovery=base/'recovery';durable=base/'results'
        rt.write(output/'protocol.json',p);enabled=[stop is not None]
        def stopping():
            if not enabled[0]:return False
            if isinstance(stop,int):
                f=output/'training_history.json'
                return f.exists() and len(rt.read(f))>=stop
            f=durable/'probes/run/calibration_update5.json'
            return f.exists() and len(rt.read(f)['rows'])>=7
        original.stopping=stopping
        def run():
            model=Tiny();writer=rt.RecoveryWriter(output,recovery,interval=99999)
            try:m.train(original,d,rt,model,None,[model.lora_A],slabs,p,shared,2.,output,recovery,writer,durable)
            finally:writer.close()
            return model
        if stop is not None:
            with self.assertRaises(SystemExit) as caught:run()
            self.assertEqual(caught.exception.code,75)
            import shutil
            shutil.rmtree(output);rt.restore(output,recovery);enabled[0]=False
        model=run();history=rt.read(output/'training_history.json');cal=rt.read(output/'calibration.json')
        self.assertEqual(len(history),25);self.assertEqual(len(cal['rows']),20)
        for i,r in enumerate(history):
            for name in m.TERMS:
                expected=(2. if name=='bands' else cal['weights'][name]) if name in arm.split('_') and (name=='bands' or i>=5) else 0.
                self.assertEqual(r['weights'][name],expected)
        self.assertEqual(cal['names'],m.names_for(arm))
        self.assertEqual({k:len(v) for k,v in cal['monitoring'].items()},{'10':5,'25':5})
        sha=rt.digest(output/'checkpoint_latest.pt');again=run()
        self.assertEqual(rt.digest(output/'checkpoint_latest.pt'),sha)
        self.assertEqual(digest(model,True),digest(again,True))
        return history,digest(model,True),cal

    def test_all_eleven_arms_run_correct_losses_with_25_update_budget(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for arm in m.ARMS:
                with self.subTest(arm=arm):self.train_fixture(Path(tmp)/arm,arm)

    def test_resume_during_prefix_calibration_and_later_updates(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for arm in ('dice_sdf','dice_bands_edge_sdf_depth'):
                expected=self.train_fixture(Path(tmp)/arm/'full',arm)
                for stop in (3,'calibration',12):
                    self.assertEqual(self.train_fixture(Path(tmp)/arm/str(stop),arm,stop),expected)

    def test_one_arm_evaluation_archive_cleanup_and_completed_skip(self):
        from scripts.colab_sam3_geofwi import parallel
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)/'bundle';durable=Path(tmp)/'results'
            rt.write(root/'bundle_manifest.json',{})
            rt.write(durable/'environment_binding.json',{})
            arm='dice_bands_edge_sdf_depth';original,slabs,p,shared=fixture(arm)
            original.stopping=lambda:False
            task=dict(id='integration_all_terms',updates=25,arm=arm,scientific_arm=arm)
            ctx=dict(prototype=p,shared=shared,bands=2.,identity=dict(binding={},extension_sha256='source',probe_source_sha256='probe'))
            loads=[];evaluations=[]
            def load(root,protocol):
                loads.append(protocol);slabs.protocol=protocol;model=Tiny()
                return original,model,None,[model.lora_A],None,None,slabs
            def evaluate(original,model,processor,volumes,p,output,root,durable):
                evaluations.append(1)
                records=[dict(case_name=c,split=s,macro_hard_dice=.7,truth_sha256=c,
                    confusion=[[1,1],[1,1]],foreground_fn=1,foreground_fp=1)
                    for s,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])] for c in cases]
                result=dict(records=records,mean_dice=dict(training_support=.7,validation=.7))
                rt.write(output/'evaluation.json',result);return result
            with patch.dict(sys.modules,{'engine':SimpleNamespace(load=load,evaluate=evaluate),'parallel':parallel}),patch.object(rt,'quota'):
                self.assertTrue(m.run_one(rt,root,durable,None,ctx,task,d))
                self.assertTrue(m.run_one(rt,root,durable,None,ctx,task,d))
            self.assertEqual(len(loads),1);self.assertEqual(len(evaluations),1)
            self.assertEqual(loads[0]['arm'],arm)
            self.assertFalse((root/'work'/task['id']/'checkpoint_latest.pt').exists())
            self.assertTrue(rt.completed(root,durable,task))

    def test_supervisor_skips_finished_and_only_dispatches_missing_arms(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)/'bundle';durable=Path(tmp)/'results'
            tasks=m.tasks_for(dict(updates=25));done={tasks[0]['id']};calls=[]
            # A later interrupted arm gets resumed first.
            f=root/'work'/tasks[-1]['id']/'checkpoint_latest.pt';f.parent.mkdir(parents=True);f.touch()
            args=SimpleNamespace(source_results=Path(tmp)/'old',previous_results=Path(tmp)/'baseline',warm_results=Path(tmp)/'warm')
            class Child:
                def __init__(self,command):
                    arm=command[-1];calls.append(arm);done.add(next(t['id'] for t in tasks if t['arm']==arm));self.returncode=0
                def poll(self):return self.returncode
            def report(*_):return dict(new_completed=len(done),new_total=11)
            with patch.object(rt,'completed',side_effect=lambda r,d,t:t['id'] in done),patch.object(m,'report',side_effect=report),patch.object(m.subprocess,'Popen',side_effect=Child):
                result=m.supervise(rt,root,durable,args,dict(tasks=tasks),lambda:False)
            self.assertEqual(result['new_completed'],11);self.assertEqual(len(calls),10)
            self.assertEqual(calls[0],tasks[-1]['arm']);self.assertEqual(set(calls),set(m.ARMS)-{m.ARMS[0]})

    def test_notebook_has_only_missing_campaign_launch_and_preserves_old_folders(self):
        import ast
        from scripts.colab_sam3_geofwi.prepare_missing import EXPORT
        nb=rt.read(EXPORT/'SAM3_GeoFWI3D_missing_constraints.ipynb')
        source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        self.assertEqual(source.count('execute([PYTHON,CAMPAIGN_SCRIPT'),1)
        for forbidden in ("campaign('preflight')",'RETRY_SCRIPT','DIAGNOSTIC_SCRIPT',"BUNDLE/'parallel.py'"):
            self.assertNotIn(forbidden,source)
        self.assertIn('results_5vol_25updates_missing11_warm5_v1',source)
        self.assertIn('--warm-results',source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
