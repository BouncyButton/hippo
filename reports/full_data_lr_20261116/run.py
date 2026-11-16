"""Execute the prespecified LR contrast against the frozen full-fold parent trainers."""
import argparse, fcntl, hashlib, json, math, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
CL=(Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())))
ST=CL/'full_data_baselines_20261115'
SL=bool(os.environ.get('SLURM_JOB_ID'))
def read(p):return json.loads(p.read_bytes())
def enc(x):return json.dumps(x,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(enc(v))
def clone(parent, dest):
    if dest.exists():return
    if parent.is_dir():shutil.copytree(parent,dest,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__'))
    elif parent.is_file():os.link(parent,dest)
def replace(p,src):
    tmp=p.with_suffix('.new');tmp.write_text(src);tmp.replace(p)
def paths(ds):
    vm=Path('/content')
    orig=ST/'mni_scratch' if SL else vm/('full_data_'+ds)
    return orig, (ST/'matched_MNI' if SL else vm/('matched_'+ds))
def scratch(k,r,root,out):
    ds=r['dataset'];orig,matched=paths(ds)
    src=(matched.parent/('native_pass_'+ds) if r['parent']=='native_pass' else matched)
    # Each lane has its own source overlay and key-specific schedules in the protocols.
    base=root/'scratch';base.mkdir(exist_ok=True)
    with (base/'prepare.lock').open('a') as g:
        fcntl.flock(g,fcntl.LOCK_EX)
        for n in ('source','entry.py','runtime.py','initialization_reference.json','paper_data_index.json','hfh_data_sha256.json','archived_hfh_schedule.py','fast_msd_storage.py','run_matched_controls.py'):
            source=src/n if (src/n).exists() else matched/n
            clone(source,base/n)
        core=base/'source/scripts/cross_dataset_campaign/capped_core.py'
        body=core.read_text()
        if 'lr_contrast_rates' not in body:
            line='        for group in optimizer.param_groups:\n            group["lr"] = lr'
            assert body.count(line)==1
            replace(core,body.replace(line,'        lr = protocol["lr_contrast_rates"][charged]\n'+line))
        folder=base/'protocols'/k
        folder.parent.mkdir(exist_ok=True);clone(src/'protocols'/k,folder)
        p=read(src/'protocols'/k/'protocol.json');p['lr_contrast_rates']=r['rates'];p['actual_peak_lr']=r['peak']
        p['lr_contrast_reference']=r['reference']
        replace(folder/'protocol.json',json.dumps(p,indent=2))
        jobs=read(base/'jobs.json') if (base/'jobs.json').exists() else {}
        jobs[k]=dict(read(src/'jobs.json')[k],protocol_sha256=sha((folder/'protocol.json').read_bytes()))
        put(base/'jobs.json',jobs)
    py=CL/'.conda/envs/hippocampus/bin/python' if SL else orig/'venv/bin/python'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[orig/'compatibility',base/'source',base/'source/scripts',base/'source/baselines/nnUNet'])),nnUNet_compile='false',WANDB_MODE='disabled')
    native=base/'work'/k;native.mkdir(parents=True,exist_ok=True)
    if not (native/'completion.json').exists():
        with (native/'training.log').open('a') as log:
            subprocess.run([str(py),'-u',str(base/'run_matched_controls.py'),'--train',k,'--original',str(orig)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    sys.path.insert(0,str(base))
    import run_matched_controls as runner
    path,receipt=runner.pack(k,native)
    trace=[json.loads(l) for l in (native/'control_exposure.jsonl').read_text().splitlines()]
    assert [x['lr'] for x in trace]==r['rates'] and sha(enc([x['case'] for x in trace]))==r['order_hash']
    assert receipt['observed']==receipt['eligible']==r['training']
    return path,native,receipt['dice']
def medical(k,r,root,out):
    ds=r['dataset'];vm=Path('/content');bundle=vm/('full_data_'+ds)
    if ds=='HFH':parent=ST/'cluster_hfh_gain';spec=parent/'protocols'/(k+'.json')
    elif ds=='MNI':parent=ST/'mni_adapter';spec=parent/'protocols'/(k+'.json')
    elif ds=='MSD':parent=vm/'standard_msd_medsam';spec=parent/'protocols'/(k+'_standard_fold.json')
    else:parent=vm/'full_data_ADNI_medical';spec=parent/'protocols'/(k+'.json')
    p=read(spec);name=k+'_lr2x'
    assert sha(enc(p['train_cases']))==r['train_hash'] and sha(enc(p['evaluation_cases']))==r['val_hash']
    assert p['learning_rate']==5e-5 and p['final_learning_rate']==5e-6
    p.update(name=name,learning_rate=1e-4,final_learning_rate=1e-5)
    if 'frozen_learning_rates' in p:p['frozen_learning_rates']=[v*2 for v in p['frozen_learning_rates']]
    target=root/'protocols'/(name+'.json');put(target,p)
    if SL:
        old=(CL/'hfh_medsam3_seed2_20261114/bundle' if ds=='HFH' else CL/'mni_medsam3_n5_f0_s0_400_20261110')
        src=old/'source';py=CL/'medsam3_runtime_20260928/venv/bin/python'
        weights=CL/'sam3_f013_s012_cluster_tail_20261022/bundle/weights/sam3.pt'
        module='scripts.colab_hfh.medsam_replication_matrix_worker' if ds=='HFH' else 'scripts.mni_medsam3_cluster.worker'
        extra=['--durable',str(root/'recovery'/name)] if ds=='HFH' else ['--record',str(out/'gpu_worker.json')]
        cmd=[str(py),'-u','-m',module,'--protocol',str(target),'--base-weights',str(weights),'--arm','dice','--output',str(out),*extra]
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[src,src/'scripts',src/'baselines/nnUNet'])),MNI_MANIFEST_SHA=sha(target.read_bytes()),HFH_BASE_WEIGHTS=str(weights),HFH_CONTROLLER_ARCHIVES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',HIPPO_GPU_WORKER='0')
    else:
        src=bundle/'source'; code=HERE
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[code,bundle/'compatibility',src,src/'scripts',src/'baselines/nnUNet'])),WANDB_MODE='disabled')
        cmd=[str(bundle/'venv/bin/python'),'-u',str(code/'covered_medical.py'),'--worker',str(target),str(root/'work'),str(root/'recovery')]
    if not (out/'completion.json').exists():
        with (out/'training.log').open('a') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    h=read(out/'training_history.json');order=read(out/'slab_schedule.json')
    assert all(math.isclose(a['lr'],b,rel_tol=1e-12,abs_tol=1e-15) for a,b in zip(h,r['rates'])) and len(h)==len(r['rates'])
    assert len(order)==2*r['updates'] and {x for x,_ in order}==set(p['train_cases'])
    if ds=='MSD':
        reference=parent/'work'/(k+'_standard_fold')/'slab_schedule.json'
        assert order==read(reference)
    else:assert sha(enc(order))==r['order_hash']
    done=read(out/'completion.json'); tr=read(out/'training_summary.json')
    assert done['status']=='complete' and done['updates']==r['updates'] and read(out/'protocol.json')==dict(p,arm='dice')
    assert tr['frozen_before']==tr['frozen_after'] and tr['adapter_before']!=tr['adapter_after']
    evalpath=out/'evaluation.json'; ev=read(evalpath)['records'];assert sha(evalpath.read_bytes())==done['evaluation_sha256']
    import filecmp
    with (out/'checkpoint_latest.pt').open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    assert h.hexdigest()==done['checkpoint_sha256']
    for split,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])]:
        records=[a for a in ev if a['split']==split]
        assert sorted(x['case_name'] for x in records)==sorted(cases)
        assert all(math.isfinite(x['macro_hard_dice']) and 'foreground_fn' in x and 'foreground_fp' in x for x in records)
    if ds=='HFH':assert [(x['case'],x['anchor']) for x in read(out/'augmentation_schedule.json')]==[tuple(x) for x in order]
    archive=root/(name+'_compact.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.suffix in ('.json','.csv','.jsonl','.log'):z.write(f,f.name)
        z.write(spec,'parent_protocol.json')
    subset=set(p.get('paired_evaluation_cases',p['evaluation_cases']))
    return archive,out,mean(a['macro_hard_dice'] for a in ev if a['split']=='validation' and a['case_name'] in subset)
def run(k,cloud):
    r=read(HERE/'plan.json')[k];assert r['status']=='new'
    root=HERE/r['dataset'];root.mkdir(exist_ok=True)
    out=root/'work'/(k+'_lr2x');out.mkdir(parents=True,exist_ok=True)
    final=root/(k+'_full_lr_v1_results.zip')
    with (out/'lr.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if cloud:
            sys.path.append('/content')
            from continue_colab import existing,publish,session
            drive=session()
            if existing(drive,final.name):return
        if not final.exists():
            builder=medical if r['model']=='medsam3' else scratch
            orig,native,dice=builder(k,r,root,out)
            receipt=dict(key=k,arm='full_lr_v1',old_peak=r['old_peak'],peak=r['peak'],updates=r['updates'],training=r['training'],paired_dice=dice,plan_sha256=sha(enc(r)))
            dest=final.with_suffix('.partial')
            shutil.copyfile(orig,dest)
            with zipfile.ZipFile(dest,'a',zipfile.ZIP_DEFLATED) as z:
                z.writestr('lr_contrast_receipt.json',enc(receipt));z.writestr('lr_contrast_plan.json',enc(r));z.write(__file__,'lr_contrast_driver.py')
            with zipfile.ZipFile(dest) as z:assert z.testzip() is None
            dest.replace(final)
        if cloud:
            proof=publish(drive,k,final)
            put(out/'durable_ack.json',dict(id=proof['id'],sha256=proof['sha256Checksum']))
            cleanup(k,r,root)
        print('VERIFIED',final.name,flush=True)
def cleanup(k,r,root):
    out=root/'work'/(k+'_lr2x');ack=read(out/'durable_ack.json'); final=root/(k+'_full_lr_v1_results.zip')
    assert sha(final.read_bytes())==ack['sha256']
    places=[out,root/'recovery'/(k+'_lr2x'),root/'scratch/work'/k,root/'scratch/recovery'/k]
    removed=[]
    for place in places:
        for f in place.rglob('*.pt'):
            if f.name in ('checkpoint_latest.pt','checkpoint_best.pt'):
                f.unlink();removed.append(str(f))
    put(out/'cleanup.json',dict(archive=ack,removed=removed))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('key');parser.add_argument('--cloud',action='store_true');parser.add_argument('--clean',action='store_true');a=parser.parse_args()
    if a.clean:
        r=read(HERE/'plan.json')[a.key];cleanup(a.key,r,HERE/r['dataset'])
    else:run(a.key,a.cloud)
