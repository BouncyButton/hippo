"""Isolated qualitative reproductions using frozen scientific training code.

No sweep, original campaign mutation, or implicit replacement of thesis scores.
Saved calibration is reused explicitly for MSD/ADNI. SAM3 retains its original
five-update prefix and adapted calibration. Predictions use image/text only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

HOME_DIR=Path('/mnt/beegfsstudents/home/3160552')
GEO=HOME_DIR/'sam3_f013_s012_cluster_tail_20261022/bundle'
ADNI=HOME_DIR/'adni_medsam3_2sample_3fold_3seed_14arms_20261018'
BASE=GEO/'weights/sam3.pt'
MEDICAL=HOME_DIR/'medsam3_runtime_20260928/weights/best_lora_weights.pt'

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_files(root,manifest):
    for rel,sha in manifest.items():
        p=root/rel
        if not p.is_file() or digest(p)!=sha:raise ValueError(f'Frozen input differs: {p}')

def setup(stage):
    root=stage/'msd'
    if not root.exists():
        assert digest(stage/'msd_bundle.tar.gz')==read(stage/'msd_payload_sha256.json')['sha256']
        with tarfile.open(stage/'msd_bundle.tar.gz') as t:
            for m in t.getmembers():
                if not m.isfile() or not m.name.startswith('msd/') or '..' in Path(m.name).parts:
                    raise ValueError('Unexpected bundle member')
            t.extractall(stage,filter='data')
    (root/'weights').mkdir(exist_ok=True)
    for name,target in [('sam3.pt',BASE),('best_lora_weights.pt',MEDICAL)]:
        p=root/'weights'/name
        if not p.exists():p.symlink_to(target)
    (stage/'runs').mkdir(exist_ok=True)
    if not (root/'work').exists():(root/'work').symlink_to(stage/'runs',target_is_directory=True)
    verify_files(root,read(root/'bundle_manifest.json'))
    verify_files(root/'data',read(root/'data_manifest.json'))
    verify_files(GEO,read(GEO/'bundle_manifest.json'))
    verify_files(GEO/'data',read(GEO/'data_manifest.json'))
    p=read(ADNI/'protocols/ADNI_f1_s2/protocol.json')
    assert digest(p['data'])==p['dataset_sha256']
    assert digest(BASE)==p['base_sha256']
    assert digest(MEDICAL)==p['lora_sha256']
    write(stage/'preflight.json',dict(status='passed',checked_at=time.time(),
        msd_bundle_sha256=digest(root/'bundle_manifest.json'),
        geo_bundle_sha256=digest(GEO/'bundle_manifest.json'),
        adni_dataset_sha256=p['dataset_sha256'],base_sha256=p['base_sha256'],
        medical_adapter_sha256=p['lora_sha256'],
        available_bytes=os.statvfs(stage).f_bavail*os.statvfs(stage).f_frsize,
        maximum_new_models=6,workers=2))

def environment(stage,task):
    env=os.environ.copy()
    root=GEO if task['dataset']=='GeoFWI3D' else stage/'msd' if task['dataset']=='MSD' else ADNI
    env['PYTHONPATH']=os.pathsep.join([str(root),str(root/'source'),str(root/'source/scripts')])
    return env,root

def volume_evaluation(stage,task,original,model,processor,p,output,loader,forward,msd=False):
    import numpy as np
    import torch
    from scripts.train_medsam3_few_shot import parameter_digest
    if hasattr(processor.transform,'clear'):
        processor.transform.clear();processor.transform.enabled=False
    model.eval().requires_grad_(False)
    identity=dict(adapter_sha256=parameter_digest(model,True),protocol_sha256=digest(output/'protocol.json'))
    if (output/'prediction_identity.json').exists():
        if read(output/'prediction_identity.json')!=identity:raise ValueError('Prediction resume identity differs')
    write(output/'prediction_identity.json',identity)
    records=read(output/'evaluation.json')['records'] if (output/'evaluation.json').exists() else []
    for split,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])]:
        for case in cases:
            target=output/'predictions'/split/(case+'.npz')
            existing=next((r for r in records if r['case']==case and r['split']==split),None)
            if existing:
                if not target.exists() or digest(target)!=existing['artifact_sha256']:
                    raise ValueError('Saved prediction artifact missing or corrupted')
                continue
            if time.time()>float(os.environ.get('HIPPO_DEADLINE','inf')):raise SystemExit(75)
            image,truth,affine=loader(case)
            assert image.shape==truth.shape and image.ndim==3
            prediction=np.empty(image.shape,dtype=np.uint8)
            with torch.inference_mode():
                for z in range(image.shape[2]):
                    pixels=np.round(image[:,:,z].T*255).astype(np.uint8)
                    logp,_=forward(pixels)
                    probs=logp.exp()
                    if not torch.isfinite(probs).all() or not torch.allclose(probs.sum(0),torch.ones_like(probs[0]),atol=1e-5):
                        raise ValueError('Invalid categorical probability field')
                    prediction[:,:,z]=probs.argmax(0).cpu().numpy().T
            n=len(p['classes'])
            matrix=np.bincount((truth.astype(np.int64)*n+prediction).ravel(),minlength=n*n).reshape(n,n)
            metrics=original.metrics(truth,prediction) if msd else original.confusion_metrics(matrix.tolist(),p['classes'])
            dice=[]
            for c in range(n):
                denominator=int(matrix[c,:].sum()+matrix[:,c].sum())
                dice.append(2*int(matrix[c,c])/denominator if denominator else 1.)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f,image=image.astype(np.float32),truth=truth.astype(np.uint8),prediction=prediction,
                    affine=np.asarray(affine) if affine is not None else np.empty((0,)),classes=np.asarray(p['classes']))
            temporary.replace(target)
            with np.load(target,allow_pickle=False) as check:
                assert np.array_equal(check['prediction'],prediction) and np.array_equal(check['truth'],truth)
            records.append(dict(case=case,case_name=case,split=split,**metrics,
                qualitative_metrics=dict(foreground_dice=float(np.mean(dice[1:])),background_dice=dice[0],
                    all_class_dice=float(np.mean(dice)),per_class_dice=dice,
                    foreground_fn=int(matrix[1:,0].sum()),foreground_fp=int(matrix[0,1:].sum()),
                    ap_swaps=int(matrix[1,2]+matrix[2,1]) if n==3 else None,
                    confusion=matrix.tolist(),empty_class_convention='Dice=1 when prediction and reference both empty'),
                prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
                artifact=str(target.relative_to(output)),artifact_sha256=digest(target),shape=list(truth.shape)))
            write(output/'evaluation.json',dict(records=records))
            print('SAVED PREDICTION',split,case,records[-1]['qualitative_metrics']['foreground_dice'],flush=True)
    score='ap_mean_dice' if msd else 'macro_hard_dice'
    result=dict(records=records,mean_dice={s:float(np.mean([r[score] for r in records if r['split']==s]))
               for s in ('training_support','validation')})
    write(output/'evaluation.json',result)
    return result

def run_geo(stage,task,output):
    import engine
    import runtime as rt
    import replication
    import missing_constraints as recipe
    import sam3_edge_diagnose_v1 as diag
    from scripts.cross_dataset_campaign.common import install_signals
    install_signals();recipe.ARMS=replication.ARMS
    p=replication.protocol(GEO,task)
    write(output/'protocol.json',p)
    original,model,processor,parameters,criterion,volumes,slabs=engine.load(GEO,p)
    shared_path=GEO/'durable/calibration/f1s2/calibration.json'
    shared=read(shared_path)
    initialization=dict(frozen=original.parameter_digest(model,False),adapter=original.parameter_digest(model,True))
    if shared['initialization']!=initialization or shared['inventory']!=slabs.inventory:
        raise ValueError('SAM3 calibration context mismatch')
    write(output/'training_inventory.json',slabs.inventory)
    write(output/'shared_calibration.json',shared)
    write(output/'runtime_recipe.json',dict(source=str(GEO),shared_calibration_sha256=digest(shared_path),
        calibration_policy='Original initialization band coefficient, new original update-five calibration for edge/depth',
        fresh_reproduction=True))
    class LocalWriter:
        def capture(self,force=False):
            # Training commits atomically to BeeGFS before calling capture.
            checkpoint=output/'checkpoint_latest.pt'
            if force and not checkpoint.exists():raise ValueError('Missing recovery checkpoint')
    recipe.train(original,diag,rt,model,criterion,parameters,slabs,p,shared,shared['weights']['bands'],
                 output,stage/'recovery'/task['id'],LocalWriter(),stage/'diagnostics')
    def loader(case):
        import nibabel as nib
        image,truth=volumes.load(case)
        affine=nib.load(Path(p['data'])/p['manifest']['cases'][case]['image']).affine
        return image,truth,affine
    result=volume_evaluation(stage,task,original,model,processor,p,output,loader,
        lambda pixels:original.forward_plane(model,processor,pixels,p))
    write(output/'completion.json',dict(status='complete',updates=25,validation_dice=result['mean_dice']['validation'],
        training_dice=result['mean_dice']['training_support'],checkpoint_sha256=digest(output/'checkpoint_latest.pt')))

def run_msd(stage,task,output):
    import engine
    import torch
    import torch._functorch.config as config
    if hasattr(config,'donated_buffer'):config.donated_buffer=False
    torch.set_num_threads(4)
    for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM):signal.signal(sig,engine.request_stop)
    root=stage/'msd';durable=stage/'msd_durable'
    # This is an explicitly identified reproduction with fixed published weights,
    # not a forged Colab calibration receipt for a different GPU environment.
    calibration=read(stage/'reference/msd_fold0_calibration.json')
    write(durable/'calibration/fold0/calibration.json',calibration)
    def reference_calibration(root_arg,durable_arg,fold):
        assert fold==0
        assert digest(durable/'calibration/fold0/calibration.json')==digest(stage/'reference/msd_fold0_calibration.json')
        return calibration
    engine.fold_calibration=reference_calibration
    original_load=engine.load
    def checked_load(root_arg,p):
        value=original_load(root_arg,p)
        original,model,processor,criterion,parameters,slabs,old=value
        write(output/'training_inventory.json',slabs.inventory)
        return value
    engine.load=checked_load
    def evaluate(original,model,processor,root_arg,p,out):
        def loader(case):
            source,image=original.load_image(root_arg/'data',case)
            return image,original.read_labels(root_arg/'data',case,source),source.affine
        return volume_evaluation(stage,task,original,model,processor,p,out,loader,
            lambda pixels:original.forward_plane(model,processor,pixels),msd=True)['records']
    engine.evaluate=evaluate
    engine.train(root,durable,task,float(os.environ.get('HIPPO_DEADLINE','inf')))

def run_adni(stage,task,output):
    import torch
    from scripts.cross_dataset_campaign import adni_foundation_full_runtime as rt
    original=rt.original
    rt.configure_retained_backward();torch.set_num_threads(4);original.install_signals()
    protocol=read(ADNI/f'protocols/ADNI_f{task["fold"]}_s{task["seed"]}/protocol.json')
    p=dict(protocol,arm=task['arm']);write(output/'protocol.json',p)
    original.reset_seed(p['seed']);model,processor=rt.load_foundation(p,BASE)
    for name,param in model.named_parameters():param.requires_grad_(original.is_adapter(name))
    parameters=[v for v in model.parameters() if v.requires_grad]
    assert sum(v.numel() for v in parameters)==18497664
    criterion=original.native_loss(model);volumes=original.Volumes(p);slabs=original.SupportSlabs(volumes,p,processor)
    expected_inventory=read(ADNI/'runs'/task['id']/'training_inventory.json')
    assert slabs.inventory==expected_inventory
    prior=read(ADNI/'runs'/task['id']/'training_summary.json')
    assert original.parameter_digest(model,False)==prior['frozen_before']
    assert original.parameter_digest(model,True)==prior['adapter_before']
    write(output/'training_inventory.json',slabs.inventory)
    calibration=read(stage/'reference'/task['id']/'calibration.json')
    assert calibration==read(ADNI/'runs'/task['id']/'calibration.json')
    write(output/'calibration.json',calibration)
    write(output/'runtime_recipe.json',dict(source=str(ADNI/'source'),fresh_reproduction=True,
        calibration_policy='Original audited coefficients, unchanged; no recalibration or hyperparameter search'))
    original.train(model,criterion,parameters,slabs,p,calibration,task['arm'],output)
    def loader(case):
        image,truth=volumes.load(case)
        return image,truth,None
    result=volume_evaluation(stage,task,original,model,processor,p,output,loader,
        lambda pixels:original.forward_plane(model,processor,pixels,p))
    write(output/'completion.json',dict(status='complete',updates=200,validation_dice=result['mean_dice']['validation'],
        training_dice=result['mean_dice']['training_support'],checkpoint_sha256=digest(output/'checkpoint_latest.pt')))

def run_one(stage,task):
    output=stage/'runs'/task['id'];output.mkdir(parents=True,exist_ok=True)
    if (output/'qualitative_complete.json').exists():return
    import torch
    write(output/'reproduction_provenance.json',dict(task=task,created_at=time.time(),original_checkpoint_recovered=False,
        new_matched_reproduction=True,replaces_original_experiment=False,
        source_driver_sha256=digest(__file__),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),device_memory=torch.cuda.get_device_properties(0).total_memory,
        calibration_reused_from_original=True,scientific_results_unchanged=True))
    {'GeoFWI3D':run_geo,'MSD':run_msd,'ADNI':run_adni}[task['dataset']](stage,task,output)
    done=read(output/'completion.json');p=read(output/'protocol.json');ev=read(output/'evaluation.json')
    expected={(split,c) for split,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])] for c in cases}
    assert {(r['split'],r['case']) for r in ev['records']}==expected and len(ev['records'])==len(expected)
    history=read(output/'training_history.json')
    assert [r['update'] for r in history]==list(range(1,task['updates']+1))
    comparison=dict(original_dice=task['expected_dice'],reproduced_dice=done['validation_dice'],
        difference_pp=100*(done['validation_dice']-task['expected_dice']),
        interpretation='New qualitative reproduction; original scores remain the thesis evidence. Review discrepancies before using panels.')
    write(output/'original_comparison.json',comparison)
    write(output/'qualitative_complete.json',dict(status='complete',task=task,volumes=len(expected),comparison=comparison))

def archive_run(stage,task):
    output=stage/'runs'/task['id'];dest=stage/'exports'/(task['id']+'.tar.gz')
    ready=dest.with_suffix('.ready.json')
    if ready.exists():return
    assert (output/'qualitative_complete.json').exists()
    files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp') and p.name!='archive_manifest.json'}
    write(output/'archive_manifest.json',dict(task=task,files=files))
    dest.parent.mkdir(exist_ok=True);temp=dest.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as tar:
        for rel in [*files,'archive_manifest.json']:tar.add(output/rel,arcname=task['id']+'/'+rel,recursive=False)
    temp.replace(dest)
    write(ready,dict(file=dest.name,sha256=digest(dest),bytes=dest.stat().st_size,run=task['id'],status='ready_for_off_cluster_archive'))

def worker(stage,rank):
    failures=[]
    for task in read(stage/'plan.json')['tasks']:
        if task['worker']!=rank:continue
        env,root=environment(stage,task)
        output=stage/'runs'/task['id'];output.mkdir(exist_ok=True,parents=True)
        with (output/'train.log').open('a') as log:
            process=subprocess.Popen([sys.executable,__file__,'one','--stage',str(stage),'--task',task['id']],env=env,stdout=log,stderr=subprocess.STDOUT)
            def stop(*_):process.send_signal(signal.SIGUSR1)
            for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
            code=process.wait()
        if code:
            failures.append(dict(task=task['id'],exit_code=code));write(stage/f'worker{rank}_failures.json',failures)
            if code==75:break
            continue
        archive_run(stage,task)
    write(stage/f'worker{rank}_complete.json',dict(status='failed' if failures else 'complete',failures=failures))
    if failures:raise SystemExit(1)

def acknowledge(stage,receipt):
    """Remove only this job's generated artifacts after local hash verification."""
    assert receipt['status']=='verified_off_cluster' and receipt['local_archive']
    name=receipt['file']
    assert Path(name).name==name and name.endswith('.tar.gz')
    archive=stage/'exports'/name
    ready=read(archive.with_suffix('.ready.json'))
    assert ready['sha256']==receipt['sha256']
    if archive.exists():assert digest(archive)==receipt['sha256']
    write(stage/'receipts'/(name+'.json'),receipt)
    tasks=read(stage/'plan.json')['tasks'];removed=[]
    def remove(p,expected=None):
        p=Path(p)
        if not p.exists():return
        assert p.resolve().is_relative_to(stage.resolve()) and not p.is_symlink()
        sha=digest(p)
        if expected is not None:assert sha==expected
        removed.append(dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=sha))
        p.unlink()
    if name!='figures.tar.gz':
        task=next(t for t in tasks if name==t['id']+'.tar.gz')
        output=stage/'runs'/task['id'];done=read(output/'completion.json')
        assert (output/'qualitative_complete.json').exists()
        remove(output/'checkpoint_latest.pt',done['checkpoint_sha256'])
        recovery=stage/'msd_durable/recovery'/task['id']
        for p in recovery.glob('recovery-*.tar'):remove(p)
    else:
        # Panel rendering has finished; every prediction must also be archived.
        for task in tasks:
            assert (stage/'receipts'/(task['id']+'.tar.gz.json')).exists()
            output=stage/'runs'/task['id']
            for record in read(output/'evaluation.json')['records']:
                remove(output/record['artifact'],record['artifact_sha256'])
    remove(archive,receipt['sha256'])
    write(stage/'cleanup'/(name+'.json'),dict(receipt=receipt,removed=removed,time=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['setup','worker','one','ack'])
    parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--task')
    parser.add_argument('--receipt-json')
    args=parser.parse_args();stage=args.stage.resolve()
    if args.mode=='setup':setup(stage)
    elif args.mode=='worker':worker(stage,int(os.environ['SLURM_PROCID']))
    elif args.mode=='ack':acknowledge(stage,json.loads(args.receipt_json))
    else:
        task=next(t for t in read(stage/'plan.json')['tasks'] if t['id']==args.task)
        _,root=environment(stage,task)
        sys.path[:0]=[str(root),str(root/'source'),str(root/'source/scripts')]
        run_one(stage,task)
