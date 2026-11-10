"""Portable package, calibration, recovery, evaluation and scheduling checks."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.colab_sam3_geofwi import runtime as rt, parallel as scheduler, fast_eval as fe
sys.path.insert(0,str(Path(__file__).parent))
from scripts.colab_sam3_geofwi import engine
from scripts.cross_dataset_campaign.common import confusion_metrics

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/sam3_geofwi_colab_exploration_20261018'
FROZEN=ROOT/'reports/sam3_geofwi_5vol_update_sweep_20261016/frozen'


class PackageTests(unittest.TestCase):
    def test_two_run_package_has_no_baseline_and_only_ten_percent_targets(self):
        import zipfile
        from scripts.colab_sam3_geofwi.prepare_two import EXPORT as two, NOTEBOOK, CODE
        plan = rt.verify_plan(two/'bundle')
        self.assertEqual(len(plan['tasks']), 2)
        self.assertEqual({t['scientific_arm'] for t in plan['tasks']}, {'dice_bands','dice_bands_edge'})
        for task in plan['tasks']:
            self.assertEqual(task['updates'],25)
            self.assertEqual(task['bands_target'],.1)
            self.assertEqual(task['edge_target'],.1 if task['scientific_arm']=='dice_bands_edge' else None)
        nb = rt.read(two/NOTEBOOK)
        for cell in nb['cells']:
            if cell['cell_type']=='code': ast.parse(cell['source'])
        source = '\n'.join(c['source'] for c in nb['cells'])
        self.assertIn('MAX_NEW_RUNS=2',source)
        self.assertIn('--baseline-results',source)
        manifest = rt.read(two/'bundle/bundle_manifest.json')
        with zipfile.ZipFile(two/CODE) as archive:
            self.assertEqual(set(archive.namelist()),set(manifest)|{'bundle_manifest.json'})
            for name,sha in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),sha)

    def test_saved_baseline_is_read_without_becoming_a_new_task(self):
        from scripts.colab_sam3_geofwi.prepare_two import EXPORT as two
        root = two/'bundle'
        task = next(t for t in rt.read(root/'reference/plan.json')['tasks'] if t['arm']=='dice' and t['updates']==25)
        expected_task = dict(task,arm='baseline',scientific_arm='dice')
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path(tmp)/'old'; durable = Path(tmp)/'new'; folder=previous/'runs'/task['id']
            protocol = rt.read(root/'reference/protocols'/task['protocol']/'protocol.json')
            rt.write(folder/'protocol.json',dict(protocol,arm='dice',data='/content/data',upstream='/content/upstream'))
            rt.write(folder/'completion.json',dict(status='complete',updates=25,validation_dice=.65))
            records=[dict(split=s,case_name=c,macro_hard_dice=.65,truth_sha256='truth')
                     for s,k in [('training_support','train_cases'),('validation','evaluation_cases')]
                     for c in protocol[k]]
            rt.write(folder/'evaluation.json',dict(records=records))
            sha = rt.read(root/'plan.json')['parent_colab_manifest_sha256']
            rt.write(previous/'environment_binding.json',dict(bundle_sha256=sha,environment={'gpu':'test'}))
            rt.write(durable/'environment_binding.json',dict(environment={'gpu':'test'}))
            rt.write(folder/'archive_manifest.json',dict(task=expected_task,bundle_sha256=sha,
                files={n:rt.digest(folder/n) for n in ['protocol.json','completion.json','evaluation.json']}))
            rt.write(folder/'cleanup.json',dict(status='complete'))
            result=rt.report(root,durable,previous)
            self.assertEqual(result['new_total'],2)
            self.assertEqual(result['new_completed'],0)
            self.assertEqual(result['reused_baseline']['validation_dice'],.65)
            self.assertFalse((durable/'runs').exists())
            new_task=rt.read(root/'plan.json')['tasks'][0]
            new_folder=durable/'runs'/new_task['id']
            rt.write(new_folder/'completion.json',dict(status='complete',updates=25,validation_dice=.65))
            rt.write(new_folder/'evaluation.json',dict(records=records))
            with patch.object(rt,'completed',side_effect=lambda r,d,t:t['id']==new_task['id']):
                result=rt.report(root,durable,previous)
            self.assertEqual(result['new_completed'],1)
            self.assertEqual(result['runs'][0]['gain_vs_native_pp'],0.)
            rt.write(folder/'completion.json',dict(status='complete',updates=25,validation_dice=.9))
            with self.assertRaisesRegex(ValueError,'archive changed'):
                rt.previous_baseline(root,durable,previous)

    def test_focused_25_update_plan_preserves_all_other_scientific_settings(self):
        from scripts.colab_sam3_geofwi.prepare25 import EXPORT as focused
        plan = rt.verify_plan(focused/'bundle')
        self.assertEqual(plan['num_runs'], 8)
        self.assertEqual({t['updates'] for t in plan['tasks']}, {25})
        self.assertEqual(sum(t['arm'] == 'baseline' for t in plan['tasks']), 1)
        self.assertEqual({t['bands_target'] for t in plan['tasks'] if t['arm'] == 'dice_bands'}, {.1, .2, .4})
        self.assertEqual({t['edge_target'] for t in plan['tasks'] if t['arm'] == 'dice_bands_edge'}, {.005, .01, .02, .05})
        # A changed schedule must fail even if the package manifest were regenerated.
        original_read = rt.read
        def altered(path):
            value = original_read(path)
            if path.name == 'protocol.json' and 'reference' not in path.parts:
                value['warmup_updates'] = 5
            return value
        with patch.object(rt, 'read', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'beyond the stopping update'):
                rt.verify_plan(focused/'bundle')

    def test_focused_notebook_and_code_zip(self):
        import zipfile
        from scripts.colab_sam3_geofwi.prepare25 import EXPORT as focused, NOTEBOOK, CODE
        nb = rt.read(focused/NOTEBOOK)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
                self.assertEqual(cell['outputs'], [])
                self.assertNotIn('results_5vol_24conditions_v1', cell['source'])
        manifest = rt.read(focused/'bundle/bundle_manifest.json')
        with zipfile.ZipFile(focused/CODE) as archive:
            self.assertEqual(set(archive.namelist()), set(manifest)|{'bundle_manifest.json'})
            for name, sha in manifest.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), sha)
        uploads = rt.read(focused/'uploads.json')
        self.assertEqual(uploads[CODE], rt.digest(focused/CODE))
        self.assertEqual(uploads['sam3_geofwi_101vol_data.zip'], rt.digest(EXPORT/'sam3_geofwi_101vol_data.zip'))

    def test_notebook_cells_and_exact_24_conditions(self):
        nb=rt.read(EXPORT/'SAM3_GeoFWI3D_efficient_exploration.ipynb')
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(cell['source']);self.assertEqual(cell['outputs'],[])
        plan=rt.read(EXPORT/'bundle/plan.json')
        self.assertEqual(len(plan['tasks']),24)
        self.assertEqual(len({t['id'] for t in plan['tasks']}),24)
        self.assertEqual([t['updates'] for t in plan['tasks'] if t['arm']=='baseline'],[10,25,50,100,200])
        expected=rt.read(FROZEN/'plan.json')['tasks']
        for task in plan['tasks']:
            original={k:v for k,v in task.items() if k!='scientific_arm'}
            original['arm']=task['scientific_arm']
            self.assertIn(original,expected)

    def test_code_and_data_zips_match_every_member_hash(self):
        import zipfile
        manifest=rt.read(EXPORT/'bundle/bundle_manifest.json')
        with zipfile.ZipFile(EXPORT/'sam3_geofwi_code.zip') as z:
            self.assertEqual(set(z.namelist()),set(manifest)|{'bundle_manifest.json'})
            for name,sha in manifest.items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),sha)
        data=rt.read(EXPORT/'bundle/data_manifest.json')
        self.assertEqual(len(data),202)
        with zipfile.ZipFile(EXPORT/'sam3_geofwi_101vol_data.zip') as z:
            self.assertEqual(set(z.namelist()),{'data/'+n for n in data})
            for name,sha in data.items():self.assertEqual(hashlib.sha256(z.read('data/'+name)).hexdigest(),sha)

    def test_transplanted_training_math_is_unchanged(self):
        old=ast.parse((FROZEN/'source/scripts/cross_dataset_campaign/foundation.py').read_text())
        new=ast.parse((Path(__file__).parent/'training_loop.py').read_text())
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        # All original expressions/calls in the optimizer computation stay exact.
        def math_nodes(node):
            targets=('lr_for_update','zero_grad','forward_slab','isfinite','backward','clip_grad_norm_','step','append')
            return [ast.dump(n,include_attributes=False) for n in ast.walk(node)
                    if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in targets)
                    or (isinstance(n.func,ast.Attribute) and n.func.attr in targets))]
        self.assertEqual(math_nodes(a),math_nodes(b))


class CalibrationTests(unittest.TestCase):
    def test_shared_gram_uses_frozen_sequential_target_rule_without_extra_forwards(self):
        source=(FROZEN/'source/scripts/cross_dataset_campaign/foundation.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='calibrate')
        def forbidden(*args):raise AssertionError('Repeated calibration forward')
        ns=dict(np=np,read=rt.read,write=rt.write,reset_seed=lambda _:None,forward_slab=forbidden,stopping=lambda:False)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_calibration','exec'),ns)
        shared=dict(names=['native','bands','edge'],rows=[dict(case='case',anchor=i,
                    gram=[[4.,.2,.1],[.2,9.+i,.3],[.1,.3,16.+i]],
                    losses=dict(native=1.,bands=2.,edge=3.)) for i in range(20)])
        slab=SimpleNamespace(calibration_schedule=lambda _: [('case',i) for i in range(20)])
        for band in (.1,.2,.4):
            for edge in (None,.005,.01,.02,.05):
                arm='dice_bands' if edge is None else 'dice_bands_edge'
                names=['native','bands']+(['edge'] if edge else [])
                p=dict(calibration_seed=1,calibration_slabs_per_volume=4,gradient_ratio_target=band,
                       gradient_ratio_p95_cap=5*band,gradient_ratio_targets=dict(bands=band,edge=edge),
                       gradient_ratio_p95_caps=dict(bands=5*band,edge=5*edge if edge else None))
                with tempfile.TemporaryDirectory() as tmp:
                    folder=Path(tmp);rows=engine.project_rows(shared,names)
                    rt.write(folder/'calibration_partial.json',dict(names=names,rows=rows))
                    cal=ns['calibrate'](None,None,None,slab,p,arm,folder)
                coefficients=[1.]
                for i,name in enumerate(names[1:],1):
                    ratios=[np.sqrt(r['gram'][i][i]/(np.array(coefficients)@np.array(r['gram'])[:i,:i]@np.array(coefficients))) for r in rows]
                    target=p['gradient_ratio_targets'][name]
                    expected=min(target/np.median(ratios),5*target/np.quantile(ratios,.95))
                    self.assertEqual(cal['weights'][name],expected);coefficients.append(expected)


class RecoveryTests(unittest.TestCase):
    def test_same_optimizer_result_and_rng_after_interrupted_resume(self):
        import contextlib,io,random,time
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.lora_A=torch.nn.Parameter(torch.tensor([.2,.4]));self.base=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        def parameter_digest(model,adapter):
            return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for n,p in model.named_parameters() if n.startswith('lora')==adapter)).hexdigest()
        def reset(seed):torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        def forward(model,*args):
            x=model.lora_A;native=(x.square()*torch.rand_like(x)).sum()
            return dict(native=native,bands=x.pow(3).sum(),edge=x.sin().square().sum(),sdf=x.sum()*0,depth=x.sum()*0)
        cpu_torch=SimpleNamespace(**{n:getattr(torch,n) for n in ['optim','nn','get_rng_state','set_rng_state','load','isfinite']},
            cuda=SimpleNamespace(get_rng_state_all=lambda:[],set_rng_state_all=lambda _:None))
        def load_function(path,stop):
            fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
            ns=dict(torch=cpu_torch,np=np,random=random,time=time,json=json,read=rt.read,write=rt.write,digest=rt.digest,
                reset_seed=reset,parameter_digest=parameter_digest,is_adapter=lambda n:n.startswith('lora'),
                stopping=stop,forward_slab=forward,lr_for_update=lambda i,h,w,a,b:a+(b-a)*i/h,
                save_checkpoint=lambda path,value:torch.save(value,path))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
            return ns['train']
        writer=SimpleNamespace(capture=lambda **kwargs:None)
        p=dict(updates=7,seed=0,learning_rate=.01,final_learning_rate=.001,warmup_updates=2,
               constraint_warmup_updates=2,learning_rate_horizon_updates=200,accumulation=2)
        slab=SimpleNamespace(schedule=lambda _: [('case',0)]*14)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);models=[];states=[]
            for name in ('original','colab','resumed'):
                out=root/name;out.mkdir();rt.write(out/'protocol.json',p)
                count=[0]
                def stop():count[0]+=1;return name=='resumed' and count[0]==4
                file=FROZEN/'source/scripts/cross_dataset_campaign/foundation.py' if name=='original' else Path(__file__).parent/'training_loop.py'
                model=Tiny();fn=load_function(file,stop)
                args=(model,None,[model.lora_A],slab,p,{'weights':{'bands':.1,'edge':0.}},'dice_bands',out)
                extra=() if name=='original' else (writer,root/(name+'_recovery'))
                if name=='resumed':
                    with self.assertRaises(SystemExit):fn(*args,*extra)
                    rt.check_ledger(out,extra[1]);model=Tiny();args=(model,None,[model.lora_A],*args[3:])
                    fn=load_function(file,lambda:False)
                fn(*args,*extra)
                models.append(model.lora_A.detach().clone())
                states.append(torch.load(out/'checkpoint_latest.pt',weights_only=False))
            for result in models[1:]:torch.testing.assert_close(result,models[0],rtol=0,atol=0)
            for state in states[1:]:
                self.assertEqual(state['history'],states[0]['history'])
                self.assertTrue(torch.equal(state['torch_rng'],states[0]['torch_rng']))

    def test_atomic_recovery_and_newer_charged_updates_are_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'work';rec=root/'drive';out.mkdir()
            for n,v in [('protocol.json',{'updates':25}),('checkpoint_ledger.json',{'step':3}),('calibration.json',{'weight':.1})]:rt.write(out/n,v)
            (out/'checkpoint_latest.pt').write_bytes(b'checkpoint3')
            writer=rt.RecoveryWriter(out,rec);writer.close(force=True)
            restored=root/'restored';rt.restore(restored,rec)
            self.assertEqual((restored/'checkpoint_latest.pt').read_bytes(),b'checkpoint3')
            rt.write(rec/'charged_ledger.json',{'step':4})
            with self.assertRaisesRegex(RuntimeError,'lost charged updates'):rt.check_ledger(restored,rec)

    def test_variable_update_progress_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rt.write(root/'protocol.json',{'updates':25})
            rt.write(root/'checkpoint_ledger.json',{'step':25});rt.write(root/'training_summary.json',{})
            self.assertEqual(scheduler.progress(rt,root)['phase'],'evaluating')
            with scheduler.SweepLock(root):
                with self.assertRaises(RuntimeError):
                    with scheduler.SweepLock(root):pass

    def test_archive_requires_full_evaluation_and_verified_copy_before_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';out=bundle/'work/run';dur=root/'drive';rec=dur/'recovery/run'
            out.mkdir(parents=True);rt.write(bundle/'bundle_manifest.json',{})
            task=dict(id='run',updates=10)
            p=dict(train_cases=['train'],evaluation_cases=['validation'])
            rt.write(out/'protocol.json',p);rt.write(out/'training_history.json',[{}]*10)
            rt.write(out/'training_summary.json',dict(frozen_before='same',frozen_after='same'))
            for n in ('calibration.json','training_inventory.json','slab_schedule.json','runtime_recipe.json'):rt.write(out/n,{})
            (out/'curves.png').write_bytes(b'curve');(out/'train.log').write_text('training log')
            checkpoint=out/'checkpoint_latest.pt';checkpoint.write_bytes(b'final adapter')
            rt.write(out/'completion.json',dict(status='complete',updates=10,checkpoint_sha256=rt.digest(checkpoint)))
            rt.write(out/'evaluation.json',dict(records=[]))
            with self.assertRaisesRegex(ValueError,'Incomplete evaluation'):rt.archive_run(bundle,out,dur,rec,task)
            self.assertTrue(checkpoint.exists())
            records=[dict(split=s,case_name=c,confusion=[[1,0],[0,1]],foreground_fn=0,foreground_fp=0)
                     for s,c in [('training_support','train'),('validation','validation')]]
            rt.write(out/'evaluation.json',dict(records=records))
            original=rt.copy_verified
            def corrupt(source,dest):
                original(source,dest)
                if source.name=='evaluation.json':dest.write_text('corrupted')
            with patch.object(rt,'copy_verified',side_effect=corrupt):
                with self.assertRaisesRegex(IOError,'readback'):rt.archive_run(bundle,out,dur,rec,task)
            self.assertTrue(checkpoint.exists())
            rt.archive_run(bundle,out,dur,rec,task)
            self.assertFalse(checkpoint.exists());self.assertTrue(rt.completed(bundle,dur,task))


class Backbone(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward_text(self,captions,input_boxes=None,additional_text=None,device='cpu'):
        self.calls+=1;return {'value':torch.tensor(2.)}


class EvaluationTests(unittest.TestCase):
    def test_text_cache_exact_and_isolated_from_training(self):
        model=torch.nn.Module();model.backbone=Backbone();model.eval()
        def forward(m,p,pixels):
            text=m.backbone.forward_text(['salt body'],device='cpu')
            result=torch.tensor(pixels.copy()).float()+text['value'];text['value'].add_(100)
            return torch.stack([result,-result]).log_softmax(0),None
        cached=fe.FeatureReuse(model,None,forward);pixels=np.ones((2,2),np.uint8)
        with self.assertRaises(RuntimeError):cached(pixels)
        with torch.inference_mode():
            expected=forward(model,None,pixels)[0]
            for _ in range(3):torch.testing.assert_close(cached(pixels)[0],expected,rtol=0,atol=0)
        self.assertEqual(cached.counts,dict(text_encodes=1,text_hits=2))
        self.assertNotIn('forward_text',vars(model.backbone))

    def test_partial_volume_stop_resume_matches_full_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'work';out.mkdir();dur=root/'drive'
            rt.write(dur/'environment_binding.json',{'test':'cpu'})
            (root/'fast_eval.py').write_text('test');(out/'checkpoint_latest.pt').write_bytes(b'checkpoint')
            p=dict(train_cases=['training'],evaluation_cases=['validation'],classes=['background','salt'])
            rt.write(out/'protocol.json',p)
            model=torch.nn.Module().eval();processor=SimpleNamespace(transform=SimpleNamespace(clear=lambda:None,enabled=True))
            image=np.ones((2,2,3),np.float32);truth=np.ones(image.shape,np.uint8)
            volumes=SimpleNamespace(load=lambda _: (image,truth))
            counter=[0]
            def stopping():counter[0]+=1;return counter[0]==3
            original=SimpleNamespace(stopping=stopping,confusion_metrics=confusion_metrics,
                forward_plane=lambda *args:(torch.tensor([[[0.,0.],[0.,0.]],[[1.,1.],[1.,1.]]]).log_softmax(0),None))
            with patch.dict(os.environ,{'HIPPO_AUDITED_FAST_EVAL':'0'}):
                with self.assertRaises(SystemExit) as error:engine.evaluate(original,model,processor,volumes,p,out,root,dur)
                self.assertEqual(error.exception.code,75)
                self.assertEqual(rt.read(out/'evaluation_partial.json')['next_slice'],1)
                original.stopping=lambda:False
                result=engine.evaluate(original,model,processor,volumes,p,out,root,dur)
            self.assertEqual(len(result['records']),2)
            self.assertEqual(result['mean_dice'],{'training_support':1.,'validation':1.})
            self.assertFalse((out/'evaluation_partial.npz').exists())


class AdaptiveTests(unittest.TestCase):
    def test_worker_count_can_exceed_two_but_respects_reserves(self):
        gpu=dict(total_gib=80.,used_gib=15.,utilization=35.,processes={},host_free_gib=150.,cpu_workers=6)
        tuner=scheduler.AdaptiveLimit(window=180)
        workers=[]
        for i in range(3):
            workers.append(scheduler.Worker(dict(id=str(i),arm='baseline'),SimpleNamespace(pid=i),None,-1000,
                dict(step=0,volumes=0),state=dict(step=1,volumes=0,phase='training')))
            tuner.observe(i*181,gpu,workers)
            for w in workers:w.state['step']+=20
            tuner.observe(i*181+180,gpu,workers)
        self.assertEqual(tuner.target,4)
        gpu['used_gib']=76
        self.assertFalse(tuner.can_launch(gpu,workers))


if __name__=='__main__':unittest.main()
