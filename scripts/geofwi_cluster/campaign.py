"""Two isolated GPU lanes; fixed recipes, capped runs and verified archive gates."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CAP = 182178126543160
MODELS = ('unetr_vit_b16', 'nnunet', 'swinunetr', 'basic_unet16')
STOP = False
CHILD = None


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def quota():
    import re
    raw = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'], text=True)
    m = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', raw)
    if not m: raise ValueError('Cannot establish personal quota')
    used, limit = map(float, m.groups())
    return dict(used_gib=used, limit_gib=limit, free_gib=limit-used, raw=raw, checked_at=time.time())


def verify(stage, data=True):
    for name, expected in read(stage/'manifest.json').items():
        assert sha(stage/name) == expected, name
    plan = read(stage/'plan.json')
    assert plan['cap'] == CAP and plan['workers'] == 2
    expected = {(m,f,s) for m in MODELS for f in plan['new_folds'] for s in range(3)}
    assert {(t['model'],t['fold'],t['seed']) for t in plan['tasks']} == expected
    assert len(plan['tasks']) == len(expected)
    splits = read(stage/'splits.json')
    assert len(splits) == 3
    assert len(set().union(*(set(s['val']) for s in splits))) == 288
    for split in splits:
        assert len(set(split['train'])) == 19 and len(set(split['val'])) == 96
        assert not set(split['train']) & set(split['val'])
    for t in plan['tasks']:
        folder = stage/'protocols'/t['id']; p = read(folder/'protocol.json')
        assert sha(folder/'dataset.json') == p['pkl_sha256']
        assert sha(folder/'splits_10patients.json') == p['splits_sha256'] == sha(stage/'splits.json')
        assert sha(folder/'nnunet_architecture.json') == p['nnunet_architecture_sha256']
        assert p['model_identity'] == t['model'] and p['seeds'] == [t['seed']] and p['folds'] == [t['fold']]
        assert p['budget_flops'] == CAP and p['batch_size'] == 1 and not p['early_stopping']
        s = p['models'][p['trainer_alias']]['objectives']['dice']
        assert s['consumed_total_flops'] <= CAP < s['consumed_total_flops']+s['flops_per_update']
        assert s['max_steps'] == t['updates'] and s['calibration_flops'] == 0
    if data:
        for name, expected_sha in read(stage/'data_manifest.json').items():
            assert sha(stage/name) == expected_sha, name
    return plan


def validate_run(stage, t):
    out = stage/'runs'/t['id']; p = read(out/'protocol.json')
    done, config = read(out/'completion.json'), read(out/'config.json')
    s = p['models'][p['trainer_alias']]['objectives']['dice']
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert (done['fold'],done['seed'],done['updates']) == (t['fold'],t['seed'],t['updates'])
    assert done['budget_flops'] == CAP and done['consumed_total_flops'] == s['consumed_total_flops'] <= CAP
    assert not config['smoke_test'] and config['model_identity'] == t['model']
    assert config['augmentation'] == t['augmentation']
    assert config['initial_lr'] == p['initial_lrs'][p['trainer_alias']]
    assert config['lr_horizon_updates'] == p['lr_horizon_updates']
    assert read(out/'splits.json') == read(stage/'splits.json')
    split = read(stage/'splits.json')[t['fold']]
    for name, cases in [('validation_latest',split['val']),('validation_best',split['val']),('training_final',split['train'])]:
        doc = read(out/(name+'.json')); scores=[]
        assert len(doc['cases']) == len(cases) and {r['case_name'] for r in doc['cases']} == set(cases)
        for r in doc['cases']:
            c=r['confusion']; assert len(c)==2 and all(len(v)==2 for v in c)
            assert all(type(v)==int and v>=0 for row in c for v in row) and sum(map(sum,c))==96**3
            score=2*c[1][1]/(2*c[1][1]+c[0][1]+c[1][0]); scores.append(score)
            assert abs(score-r['macro_hard_dice']) < 1e-10
        assert abs(sum(scores)/len(scores)-doc['summary']['macro_hard_dice']) < 1e-10
    init=read(out/'initialization.json')['model_state_sha256']
    if t['model']=='unetr_vit_b16': assert init==p['pilot_initialization_sha256']
    assert init==p['reference_initializations'][str(t['seed'])]
    for other in read(stage/'plan.json')['tasks']:
        previous=stage/'runs'/other['id']/'initialization.json'
        if other['model']==t['model'] and other['seed']==t['seed'] and previous.exists():
            assert read(previous)['model_state_sha256']==init
    return dict(**t, final_dice=read(out/'validation_latest.json')['summary']['macro_hard_dice'],
        best_dice=done['best_validation_hard_dice'],training_dice=done['final_training_hard_dice'],
        total_flops=done['consumed_total_flops'],initialization_sha256=init)


def storage(stage):
    sys.path.insert(0,str(stage/'bundles/unetr/source'))
    from scripts.cross_dataset_campaign import storage as module
    return module


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None: CHILD.send_signal(signal.SIGUSR1)


def execute(cmd, env, log):
    global CHILD
    with log.open('a') as stream:
        CHILD = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        code = CHILD.wait(); CHILD = None
    if code not in (0,75,76): raise RuntimeError(f'Worker failed ({code}); see {log}')
    return code


def lane(stage):
    lane_id=int(os.environ['SLURM_PROCID']); job=os.environ['SLURM_JOB_ID']
    plan=verify(stage, data=False); archive=storage(stage)
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    import torch
    import importlib.metadata
    assert torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    expected=read(stage/'bundles/unetr/control_environment.json')
    versions={k:importlib.metadata.version(k) for k in expected['packages']}
    assert versions==expected['packages'], versions
    assert torch.version.cuda==expected['cuda'] and torch.backends.cudnn.version()==expected['cudnn']
    gpu=torch.cuda.get_device_properties(0)
    probe=stage/'gpu_probe'/job/f'{lane_id}.json'
    write(probe,dict(uuid=str(gpu.uuid),gpu=gpu.name,visible_devices=1,packages=versions,job=job))
    deadline=time.time()+120
    while not all((probe.parent/f'{i}.json').exists() for i in range(2)):
        if time.time()>deadline or STOP: raise RuntimeError('GPU isolation probe timeout')
        time.sleep(1)
    assert len({read(probe.parent/f'{i}.json')['uuid'] for i in range(2)})==2
    state=stage/f'lane{lane_id}.json'
    for t in plan['tasks'][lane_id::2]:
        out=stage/'runs'/t['id']; out.mkdir(parents=True,exist_ok=True)
        if (out/'cleanup.json').exists() and read(out/'cleanup.json')['status']=='complete':
            archive.verify_archive(out,read(out/'archive_manifest.json')); continue
        if STOP or (stage/'STOP').exists(): write(state,dict(status='paused')); return 75
        usage=quota(); write(stage/f'quota_lane{lane_id}.json',usage)
        if usage['free_gib'] < 24: raise RuntimeError('Less than 24 GiB personal quota headroom')
        child=stage/'bundles'/t['bundle']; folder=stage/'protocols'/t['id']
        env=dict(os.environ, PYTHONPATH=':'.join(str(child/p) for p in ('source','source/scripts','source/baselines/nnUNet')),
                 MPLCONFIGDIR=str(stage/f'mpl{lane_id}'),WANDB_MODE='disabled',HIPPO_COLAB_RECOVERY=str(stage/'recovery'/t['id']))
        cmd=[sys.executable,'-u',str(stage/'worker_entry.py'),str(child),'--protocol-dir',str(folder),'--output',str(out),'--seed',str(t['seed']),'--fold',str(t['fold'])]
        if not (out/'completion.json').exists():
            pf=stage/'preflight'/job/f"{lane_id}_{t['model']}"
            if not (pf/'gpu_preflight.json').exists():
                pf.mkdir(parents=True,exist_ok=True); pf_cmd=cmd.copy(); pf_cmd[pf_cmd.index('--output')+1]=str(pf)
                write(state,dict(status='preflight',task=t))
                assert execute(pf_cmd+['--preflight'],env,pf/'preflight.log')==0
                assert read(pf/'gpu_preflight.json')['status']=='passed'
            for name, src in [('protocol.json',folder/'protocol.json'),('splits.json',stage/'splits.json')]:
                target=out/name
                if target.exists(): assert sha(target)==sha(src)
                else: target.write_bytes(src.read_bytes())
            write(state,dict(status='training',task=t))
            code=execute(cmd,env,out/'train.log')
            if code: write(state,dict(status='paused',task=t)); return 75
        row=validate_run(stage,t)
        archive.prepare(out,t)
        write(state,dict(status='waiting_for_verified_archive',task=t,result=row))
        began=time.monotonic()
        while not (out/'archive_acknowledgment.json').exists():
            if STOP or time.monotonic()-began>300:
                write(state,dict(status='waiting_for_archive_offline',task=t)); return 75
            time.sleep(2)
        ack=read(out/'archive_acknowledgment.json')
        assert ack['source_manifest_sha256']==sha(stage/'manifest.json')
        archive.retire(out)
        recovery=stage/'recovery'/t['id']; marker=recovery/'latest.json'
        deleted=[]
        if marker.exists():
            info=read(marker); path=recovery/info['file']
            assert path.parent.resolve()==recovery.resolve() and not path.is_symlink()
            if path.exists():
                assert sha(path)==info['sha256']; deleted.append(dict(path=str(path),bytes=path.stat().st_size)); path.unlink()
            marker.unlink()
        (recovery/'charged_ledger.json').unlink(missing_ok=True)
        write(out/'recovery_cleanup.json',dict(status='complete',deleted=deleted))
        write(out/'result.json',row)
    write(state,dict(status='complete')); return 0


def summarize(stage, status):
    plan=read(stage/'plan.json'); rows=list(plan['reused_rows'])
    for t in plan['tasks']:
        out=stage/'runs'/t['id']
        if (out/'cleanup.json').exists() and read(out/'cleanup.json')['status']=='complete': rows.append(validate_run(stage,t))
    keys={(r['model'],r['fold'],r['seed']) for r in rows}
    assert len(keys)==len(rows) and all(math.isfinite(r['final_dice']) for r in rows)
    full=keys=={(m,f,s) for m in MODELS for f in range(3) for s in range(3)}
    passed=full and all(r['final_dice']>.5 for r in rows)
    doc=dict(status=status,rows=rows,required_cells=36,qualified=passed,complete_matrix=full,threshold=.5,updated_at=time.time())
    write(stage/'status.json',doc)
    if full:
        write(stage/'FINISHED.json',doc)
        if passed: write(stage/'QUALIFIED.json',doc)
    return doc


def run(stage):
    global CHILD
    verify(stage)
    with (stage/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
        summarize(stage,'running')
        cmd=['srun','--exclusive','--exact','--nodes=1','--ntasks=2','--cpus-per-task=4','--mem=80G',
             '--gres=gpu:4g.40gb:2','--gpus-per-task=4g.40gb:1','--gpu-bind=single:1','--kill-on-bad-exit=1',
             sys.executable,'-u',str(stage/'campaign.py'),'lane','--stage',str(stage)]
        CHILD=subprocess.Popen(cmd); code=CHILD.wait(); CHILD=None
        doc=summarize(stage,'complete' if code==0 else 'needs_resume' if code==75 else 'failed')
        if code==0: assert doc['complete_matrix']
        return code


def submit(stage):
    plan=verify(stage); receipt=stage/'submission.json'
    for child in ('unetr','basic'):
        audit=read(stage/f'cpu_audit_{child}.json')
        assert audit['status']=='passed' and audit['updates']==0
    with (stage/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists(): return read(receipt)
        rows=[r.split('|') for r in subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i|%T|%j'],text=True).splitlines()]
        own=[r for r in rows if r[2]=='hippo-geofwi-3f3s']
        if own:
            assert len(own)==1
            result=dict(status='submitted',job_id=own[0][0],recovered_from_queue=True,time=time.time())
            write(receipt,result); return result
        if (stage/'submission_intent.json').exists():
            raise RuntimeError('Prior submission outcome unknown: inspect sacct before retry; no duplicate submitted')
        if any(r[1] not in ('RUNNING','PENDING','COMPLETING') for r in rows): raise RuntimeError('Unexpected queue state')
        if any(r[1]=='PENDING' for r in rows) or len(rows)>1:
            return dict(status='waiting_for_queue_slot',queue=rows)
        usage=quota()
        assert usage['free_gib']>=24
        cmd=['sbatch','--parsable',f'--chdir={stage}',f'--output={stage}/slurm-%j.out',f'--error={stage}/slurm-%j.err']
        if rows: cmd.append('--dependency=afterany:'+rows[0][0])
        cmd += [str(stage/'run.sbatch'),str(stage)]
        write(stage/'submission_intent.json',dict(command=cmd,time=time.time()))
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0]; assert job.isdigit()
        result=dict(status='submitted',job_id=job,command=cmd,quota=usage,manifest_sha256=sha(stage/'manifest.json'),time=time.time())
        write(receipt,result); return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['verify','run','lane','submit','status']);ap.add_argument('--stage',type=Path,required=True)
    a=ap.parse_args();stage=a.stage.resolve()
    if a.command=='verify': print(json.dumps(dict(status='passed',tasks=len(verify(stage)['tasks']))))
    elif a.command=='submit': print(json.dumps(submit(stage)))
    elif a.command=='status': print(json.dumps(summarize(stage,'inspected')))
    elif a.command=='lane': sys.exit(lane(stage))
    else: sys.exit(run(stage))


if __name__=='__main__': main()
