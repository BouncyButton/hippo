"""Fill the established 14-condition seed-0 matrix, up to three independent workers."""
from collections import deque
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
import shutil

from scripts.colab_hfh import medsam_adaptive_band as band
from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_adaptive_band_campaign import validator
from scripts.colab_hfh.medsam_component_constraints_campaign import portable_gpu_sample
from scripts.colab_hfh.medsam_constraints_policy import ARMS
from scripts.colab_hfh.medsam_constraints_drive import client
from scripts.colab_hfh.medsam_seed0_matrix_worker import check_protocol, validate
from scripts.colab_hfh.runtime import read, write, digest

ROOT = Path('/content/hippo_hfh')
SOURCE = ROOT/'bundle/source'
NAME = 'medsam_gain_adaptive_matrix125_seed0_v1'
WORK = ROOT/NAME
REMOTE = 'hfh_storage:results_'+NAME
SOURCES = ('medsam_seed0_matrix_worker.py','medsam_seed0_matrix_campaign.py',
           'medsam_adaptive_band.py','medsam_adaptive_band_worker.py','medsam_adaptive_band_recovery.py')


def finish(output, protocol, control, control_path, job):
    arm = job['arm']
    paired = validator()
    paired.__globals__['validate_adaptive'] = validate
    row = paired(output,protocol,control,arm,protocol['constraint_recipe'])
    done = read(output/'completion.json')
    assert digest(output/'checkpoint_latest.pt')==done['checkpoint_sha256']
    assert read(output/'augmentation_schedule.json')==serial.zip_read(control_path,'augmentation_schedule.json')
    assert all(read(output/'loss_logging_preflight.json')[k] for k in ('exact_rng','exact_loss','exact_gradients'))
    assert read(output/'runtime_validation.json')['deterministic_algorithms']
    truth={(r['split'],r['case_name']):r['truth_sha256'] for r in control['evaluation']['records']}
    for r in read(output/'evaluation.json')['records']:
        assert truth[r['split'],r['case_name']]==r['truth_sha256']
    row['hardware']=read(output/'execution_hardware.json')
    if 'bands' in arm:
        row['band_weights']=[c['weight'] for c in read(output/'adaptive_band_state.json')['calibrations']]
    write(output/'result.json',row)
    files=[p for p in output.iterdir() if p.suffix in ('.json','.csv','.log') and p.name!='archive_manifest.json']
    write(output/'archive_manifest.json',{p.name:digest(p) for p in files})
    archive=WORK/'archives'/(job['name']+'.zip')
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[output/'archive_manifest.json']:
            z.write(p,'results/'+p.name)
        for name in SOURCES:
            z.write(Path(__file__).with_name(name),'source/'+name)
    assert archives.verify_zip(archive)==row
    client().upload(archive,REMOTE+'/compact_archives/'+archive.name)
    return row


def run():
    import torch
    api=client()
    WORK.mkdir(exist_ok=True)
    lock=(ROOT/'work_medsam3/controller.lock')
    lock.parent.mkdir(exist_ok=True)
    handle=lock.open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_bf16_supported()
    gpu=portable_gpu_sample()
    assert 'A100' in gpu['name'] and gpu['total_gib']>=70
    assert min(int(api.quota['limit'])-int(api.quota['usage']),shutil.disk_usage(WORK).free)>12*2**30
    reused=read(WORK/'reused.json')
    templates=read(WORK/'protocol_templates.json')
    old='hfh_storage:results_medsam3_gain125_3fold3seed_v1/compact_archives/'
    adaptive='hfh_storage:results_medsam_adaptive_band_seed0_v1/compact_archives/'
    controls,paths,results={},{},{}
    for name,entry in reused.items():
        f,arm=entry['fold'],entry['arm']
        p=WORK/'reused'/name
        if f or name.startswith('adaptive'):
            api.download((adaptive if name.startswith('adaptive') else old)+name,p)
        assert digest(p)==entry['sha256']
        row=archives.verify_zip(p)
        q=serial.zip_read(p,'protocol.json')
        assert (q['fold'],q['seed'],q['arm'],q['augmentation']['component'],row['updates'])==(f,0,arm,'gain',125)
        if 'bands' in arm:
            assert q['adaptive_band']==band.POLICY
        if arm=='dice':
            controls[f]=serial.archived_control(p)
            paths[f]=p
        results[f,arm]=dict(fold=f,arm=arm,validation_dice=row['validation_dice'],origin='reused',archive=name,sha256=digest(p))
    assert len(controls)==3 and len(results)==10 and len(ARMS)==14
    sources={n:digest(Path(__file__).with_name(n)) for n in SOURCES}
    api.resolve(REMOTE+'/compact_archives',create=True)
    inventory={x['name'] for x in api.children(api.resolve(REMOTE+'/compact_archives')['id'])}
    pending=deque()
    # Matched SDF-only cells first to resolve the incremental contribution of BAND.
    order=['dice_sdf']+[a for a in ARMS if a!='dice_sdf']
    for arm in order:
        for fold in range(3):
            if (fold,arm) in results:
                continue
            p=deepcopy(templates[f'dice_bands__f{fold}_s0'])
            p.update(requested_arms=[arm],adaptive_band=band.POLICY,matrix_source_hashes=sources,
                     baseline_archive_sha256=controls[fold]['archive_sha256'],
                     constraint_recipe=dict(name='gain_adaptive_matrix_v1',target=.01,ramp=20,numerics='strict_empty_grid_v1'),
                     gradient_ratio_targets=dict(bands=.01,edge=.1,depth=.1,sdf=.1),
                     calibration_policy='BAND at activation; SDF on 80 gain slabs at initialization; EDGE/DEPTH with established 40-slab calibration',
                     comparison='Seed 0, gain only, 125 updates, original 200-update LR horizon')
            check_protocol(p)
            for k in ('frozen_sample_schedule','frozen_learning_rates','base_sha256','lora_sha256','train_cases','evaluation_cases','augmentation'):
                assert p[k]==controls[fold]['protocol'][k], (fold,k)
            name=f'matrix125__{arm}__f{fold}_s0'
            job=dict(fold=fold,arm=arm,name=name)
            path=WORK/'protocols'/(name+'.json')
            if path.exists():assert read(path)==p
            write(path,p)
            if name+'.zip' in inventory:
                archive=WORK/'archives'/(name+'.zip')
                api.download(REMOTE+'/compact_archives/'+archive.name,archive)
                row=archives.verify_zip(archive)
                assert serial.zip_read(archive,'protocol.json')==dict(p,arm=arm)
                results[fold,arm]=dict(**job,validation_dice=row['validation_dice'],origin='matrix archive')
            else:
                pending.append(job)
    assert len(pending)+len(results)==42
    state=dict(hardware=gpu['name'],seed=0,updates=125,requested_workers=3,reused=len(results))
    active={}
    def publish(status):
        workers=[]
        for name,(proc,job) in active.items():
            output=WORK/'runs'/name
            ledger=output/'checkpoint_ledger.json'
            workers.append(dict(**job,pid=proc.pid,step=read(ledger)['step'] if ledger.exists() else 0,
                                phase='evaluation' if (output/'training_summary.json').exists() else 'training_or_calibration'))
        state.update(status=status,observed_at=time.time(),workers=workers,completed=len(results),
                     results=list(results.values()),remaining=len(pending),next_job=pending[0] if pending else None)
        write(WORK/'status.json',state)
        api.upload(WORK/'status.json',REMOTE+'/status.json')
        print('MATRIX_STATUS',status,len(results),[(w['name'],w['step']) for w in workers],flush=True)
    publish('starting')
    interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        last=0
        while pending or active:
            if interrupted or (WORK/'STOP').exists():break
            for name,(proc,job) in list(active.items()):
                if proc.poll() is None:continue
                output=WORK/'runs'/name
                del active[name]
                if proc.returncode:
                    raise RuntimeError(name+' failed: '+(output/'worker.log').read_text()[-2500:])
                fold=job['fold']
                row=finish(output,read(WORK/'protocols'/(name+'.json')),controls[fold],paths[fold],job)
                results[fold,job['arm']]=dict(**job,validation_dice=row['validation_dice'],delta_pp=row['delta_dice_pp'],origin='matrix')
                last=0
            while pending and len(active)<3:
                job=pending.popleft()
                name=job['name']
                output=WORK/'runs'/name
                output.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,HFH_CONTROLLER_ARCHIVES='1',MNI_MANIFEST_SHA=digest(ROOT/'bundle/manifest.json'),
                         CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                cmd=[sys.executable,'-m','scripts.colab_hfh.medsam_seed0_matrix_worker',
                     '--protocol',str(WORK/'protocols'/(name+'.json')),'--output',str(output),
                     '--durable',str(Path('/content/hfh_storage/results_'+NAME)/'runs'/name),
                     '--base-weights',str(ROOT/'sam3.pt'),'--arm',job['arm']]
                with (output/'worker.log').open('a') as log:
                    proc=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[name]=(proc,job)
                last=0
            if time.monotonic()-last>=100:
                publish('running')
                last=time.monotonic()
            if active:time.sleep(5)
        if not interrupted and not (WORK/'STOP').exists():
            assert len(results)==42
            publish('complete')
    except BaseException as e:
        state['error']=str(e)[-3000:]
        raise
    finally:
        for name,(proc,job) in active.items():
            while proc.poll() is None and not (WORK/'runs'/name/'supervisor_ready.json').exists():time.sleep(2)
            if proc.poll() is None:proc.send_signal(signal.SIGUSR1)
        for proc,job in active.values():proc.wait()
        active.clear()
        if len(results)!=42:publish('error' if 'error' in state else 'paused')


if __name__=='__main__':
    run()
