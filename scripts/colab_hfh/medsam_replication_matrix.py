"""Schedule the 14 HFH objectives with matched seed-specific native controls."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_125_campaign import local_control, validator125
from scripts.colab_hfh.medsam_adaptive_band import POLICY
from scripts.colab_hfh.medsam_adaptive_band_campaign import validator
from scripts.colab_hfh.medsam_constraints_policy import ARMS
from scripts.colab_hfh.medsam_replication_matrix_worker import check_protocol
from scripts.colab_hfh.medsam_seed0_matrix_worker import validate
from scripts.cross_dataset_campaign.common import read, write, digest

SOURCES = ('medsam_replication_matrix.py','medsam_replication_matrix_worker.py','medsam_seed0_matrix_worker.py',
           'medsam_adaptive_band.py','medsam_adaptive_band_recovery.py','medsam_matrix_acceleration.py',
           'medsam_gain_replication.py','medsam_gain_replication_worker.py','medsam_adaptive_band_worker.py')
MATCH = ('seed','fold','updates','augmentation','training_numerics','base_sha256','lora_sha256',
         'dataset_sha256','train_cases','evaluation_cases','frozen_sample_schedule','frozen_learning_rates')


def verify_cell(path, template, arm):
    row = archives.verify_zip(path)
    p = serial.zip_read(path,'protocol.json')
    if (row['arm'] != arm or row['updates'] != 125 or p['arm'] != arm
            or any(p[key] != template[key] for key in MATCH)):
        raise ValueError('Archive incompatible with requested cell: '+path.name)
    if 'bands' in arm and p.get('adaptive_band') != POLICY:
        raise ValueError('Static BAND is not the requested recipe')
    return row


def finish(output, archive, p, arm, native):
    if arm == 'dice':
        control = dict(local_control(output), archive_sha256=None)
        check = validator125()
    else:
        control = serial.archived_control(native)
        check = validator()
        check.__globals__['validate_adaptive'] = validate
        if read(output/'augmentation_schedule.json') != serial.zip_read(native,'augmentation_schedule.json'):
            raise ValueError('Paired gain schedule differs')
    row = check(output,p,control,arm,p['constraint_recipe'])
    labels = {(r['split'],r['case_name']):r['truth_sha256'] for r in control['evaluation']['records']}
    for r in read(output/'evaluation.json')['records']:
        assert labels[r['split'],r['case_name']]==r['truth_sha256']
    assert digest(output/'checkpoint_latest.pt') == read(output/'completion.json')['checkpoint_sha256']
    assert read(output/'runtime_validation.json')['deterministic_algorithms']
    assert all(read(output/'loss_logging_preflight.json')[k] for k in ('exact_rng','exact_loss','exact_gradients'))
    row['hardware'] = read(output/'execution_hardware.json')
    write(output/'result.json',row)
    files = [f for f in output.iterdir() if f.suffix in ('.log','.csv','.json') and f.name!='archive_manifest.json']
    write(output/'archive_manifest.json',{f.name:digest(f) for f in files})
    temp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in [*files,output/'archive_manifest.json']:
            z.write(f,'results/'+f.name)
        for name in SOURCES:
            z.write(Path(__file__).with_name(name),'source/'+name)
    assert archives.verify_zip(temp)==row
    temp.replace(archive)
    return row


def run(root, seed):
    from scripts.colab_hfh.medsam_constraints_drive import client
    cluster = seed == 2
    name = f'medsam_gain_adaptive_matrix125_seed{seed}_v1'
    work = root/name
    work.mkdir(parents=True,exist_ok=True)
    lock = (root/'work_medsam3/controller.lock')
    lock.parent.mkdir(exist_ok=True)
    handle = lock.open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bank = work/'archives'
    bank.mkdir(exist_ok=True)
    api = None if cluster else client()
    remote = 'hfh_storage:results_'+name
    old = 'hfh_storage:results_medsam3_gain125_3fold3seed_v1/compact_archives'
    if api:
        import torch
        if 'A100' not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory<70*2**30:
            raise ValueError('Current Colab runtime does not fit three workers')
        for folder in ('runs','compact_archives'):
            api.resolve(remote+'/'+folder,create=True)
        prior = {v['name'] for v in api.children(api.resolve(old)['id'])}
        for fold in range(3):
            for arm in ('dice','dice_edge_depth'):
                file = f'gain125__{arm}__f{fold}_s{seed}.zip'
                if file in prior:
                    api.download(old+'/'+file,bank/file)
        for file in api.children(api.resolve(remote+'/compact_archives')['id']):
            if file['name'].endswith('.zip'):
                api.download(remote+'/compact_archives/'+file['name'],bank/file['name'])
    templates = read(work/'protocol_templates.json')
    sources = {f:digest(Path(__file__).with_name(f)) for f in SOURCES}
    results, controls, active = {}, {}, {}
    for archive in sorted(bank.glob('*.zip')):
        q = serial.zip_read(archive,'protocol.json')
        f,arm = q['fold'],q['arm']
        template = templates[f'dice__f{f}_s{seed}']
        row = verify_cell(archive,template,arm)
        if arm not in ARMS or (f,arm) in results: raise ValueError('Duplicate or unknown objective')
        results[f,arm]=row
        if arm=='dice': controls[f]=archive
        if api and archive.name.startswith('matrix'):
            api.upload(archive,remote+'/compact_archives/'+archive.name)
    order = ['dice','dice_sdf','dice_sdf_depth','dice_bands_sdf_depth','dice_edge_depth',
             'dice_bands_edge_depth']+[a for a in ARMS if a not in
             ('dice','dice_sdf','dice_sdf_depth','dice_bands_sdf_depth','dice_edge_depth','dice_bands_edge_depth')]
    todo=[(fold,arm) for arm in order for fold in range(3) if (fold,arm) not in results]
    assert len(todo)+len(results)==42
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1): signal.signal(sig,stop)
    def publish(status,error=None):
        workers=[]
        for (fold,arm),(proc,out,p) in active.items():
            ledger=out/'checkpoint_ledger.json'
            workers.append(dict(fold=fold,arm=arm,pid=proc.pid,step=read(ledger)['step'] if ledger.exists() else 0))
        state=dict(status=status,seed=seed,observed_at=time.time(),workers=workers,completed=len(results),
                   remaining=len(todo),results=list(results.values()),error=error)
        write(work/'status.json',state)
        if api: api.upload(work/'status.json',remote+'/status.json')
        print('MATRIX',seed,status,len(results),workers,error or '',flush=True)
    base = Path(os.environ.get('HFH_BASE_WEIGHTS',root/'sam3.pt'))
    last=0
    try:
        while todo or active:
            if stopping or (work/'STOP').exists():break
            for key,(proc,out,p) in list(active.items()):
                if proc.poll() is None:continue
                if proc.returncode:raise RuntimeError(out.name+' '+(out/'worker.log').read_text()[-3200:])
                fold,arm=key
                archive=bank/(out.name+'.zip')
                row=finish(out,archive,p,arm,controls.get(fold))
                if api:api.upload(archive,remote+'/compact_archives/'+archive.name)
                results[key]=row
                if arm=='dice': controls[fold]=archive
                del active[key]
                last=0
            for fold,arm in todo.copy():
                if len(active)>=(2 if cluster else 3):break
                if arm!='dice' and fold not in controls:continue
                template=templates[f'dice__f{fold}_s{seed}']
                p=deepcopy(template)
                if arm!='dice':
                    p.update(constraint_recipe=dict(name='gain_adaptive_matrix_v1',target=.01,ramp=20,numerics='strict_empty_grid_v1'),
                             baseline_archive_sha256=digest(controls[fold]),gradient_ratio_targets=dict(bands=.01,sdf=.1,edge=.1,depth=.1))
                p.update(requested_arms=[arm],adaptive_band=POLICY,matrix_source_hashes=sources,
                         comparison=f'Paired seed {seed}, gain only, 125 updates, original 200-update LR horizon')
                if arm!='dice':p['calibration_policy']='BAND at activation; SDF on 80 gain slabs; EDGE/DEPTH on 40 training slabs'
                if cluster:
                    for k in ('upstream','lora_weights','data'):
                        p[k]=str(root/Path(p[k]).relative_to('/content/hippo_hfh'))
                check_protocol(p)
                task=f'matrix125__{arm}__f{fold}_s{seed}'
                path=work/'protocols'/(task+'.json')
                if path.exists() and read(path)!=p:raise ValueError('Recovery protocol changed: '+task)
                write(path,p)
                out=work/'runs'/task
                out.mkdir(parents=True,exist_ok=True)
                dur=(work/'recovery_locations' if cluster else Path('/content/hfh_storage')/('results_'+name)/'runs')/task
                env=dict(os.environ,HFH_CONTROLLER_ARCHIVES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',
                         MNI_MANIFEST_SHA=digest(root/'bundle/manifest.json'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                prefix=(['srun','--exclusive','--exact','-N1','-n1','-c4','--mem=40G','--gres=gpu:1',
                         '--gpus-per-task=1','--gpu-bind=single:1','--kill-on-bad-exit=1'] if cluster else [])
                command=[*prefix,sys.executable,'-m','scripts.colab_hfh.medsam_replication_matrix_worker',
                         '--protocol',str(path),'--output',str(out),'--durable',str(dur),'--base-weights',str(base),'--arm',arm]
                with (out/'worker.log').open('a') as log:
                    proc=subprocess.Popen(command,cwd=root/'bundle/source',env=env,stdout=log,stderr=subprocess.STDOUT)
                active[fold,arm]=(proc,out,p)
                todo.remove((fold,arm))
                last=0
            if time.monotonic()-last>=120:
                publish('running')
                last=time.monotonic()
            if active:time.sleep(6)
        publish('complete' if len(results)==42 else 'paused')
    except BaseException as e:
        publish('error',str(e))
        raise
    finally:
        for proc,out,_ in active.values():
            if proc.poll() is None:
                while proc.poll() is None and not (out/'supervisor_ready.json').exists():time.sleep(2)
                if proc.poll() is None:proc.send_signal(signal.SIGUSR1)
        for proc,_,_ in active.values():proc.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(1,2),required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.seed)
