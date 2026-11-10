"""Sequential, auditable GeoFWI3D seed screening; no cluster submission."""
import argparse
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# Packaged beside this module from the existing audited Colab storage implementation.
try:
    import storage_runtime as io
except ImportError:
    from scripts.colab_msd import runtime as io

MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')
CAP = 182178126543160
THRESHOLD = .5
read, write, digest = io.read, io.write, io.digest


def task(seed, fold, model):
    return dict(id=f'GeoFWI3D_{model}_baseline_f{fold}s{seed}', kind='conventional',
                dataset='GeoFWI3D', model=model, fold=fold, seed=seed, arm='dice',
                protocol=f'{model}_f{fold}_s{seed}')


def tasks(seed):
    return [task(seed, f, m) for f in range(3) for m in MODELS]


def decide(seed, rows):
    """Strictly greater than 50%, for every model/fold, never an average gate."""
    expected = tasks(seed)
    by_id = {}
    for row in rows:
        if row['seed'] != seed:
            continue
        if row['id'] in by_id:
            raise ValueError('Duplicate result')
        if not math.isfinite(row['final_dice']) or not 0 <= row['final_dice'] <= 1:
            raise ValueError('Invalid score is a technical failure, not seed rejection')
        by_id[row['id']] = row
    if set(by_id) - {t['id'] for t in expected}:
        raise ValueError('Unexpected model/fold identity')
    for t in expected:
        r = by_id.get(t['id'])
        if r and r['final_dice'] <= THRESHOLD:
            return dict(seed=seed, status='rejected', failed_run=t['id'],
                        failed_dice=r['final_dice'], completed_cells=len(by_id),
                        unrun_cells=[x['id'] for x in expected if x['id'] not in by_id])
    missing = [t for t in expected if t['id'] not in by_id]
    if missing:
        return dict(seed=seed, status='pending', next_task=missing[0], completed_cells=len(by_id))
    for model in MODELS:
        if len({by_id[t['id']]['initialization_sha256'] for t in expected if t['model']==model}) != 1:
            raise ValueError('Initialization changed across folds')
    return dict(seed=seed, status='qualified', completed_cells=12,
                minimum_dice=min(r['final_dice'] for r in by_id.values()))


def summary(rows):
    decisions, qualified = [], []
    seed = 0
    while True:
        d = decide(seed, rows)
        decisions.append({k:v for k,v in d.items() if k != 'next_task'})
        if d['status'] == 'qualified':
            qualified.append(seed)
        if len(qualified) == 3 or d['status'] == 'pending':
            break
        seed += 1
    return dict(status='complete' if len(qualified)==3 else 'in_progress',
                qualified_seeds=qualified, required_seeds=3,
                completed_cells=len(rows), threshold=THRESHOLD, comparison='strictly_greater',
                completed_training_flops=sum(r.get('training_flops',0) for r in rows),
                selection_metric='final_budget_mean_per_volume_foreground_hard_dice',
                decisions=decisions, runs=rows,
                interpretation='Validation-selected seed cohort; retain all rejected seeds. Not an unbiased random-seed estimate or independent test evaluation.')


def verify(root):
    for name, sha in read(root/'bundle_manifest.json').items():
        if digest(root/io.safe_member(name)) != sha:
            raise ValueError('Code bundle changed: '+name)
    for name, sha in read(root/'data_manifest.json').items():
        if digest(root/io.safe_member(name)) != sha:
            raise ValueError('Data changed: '+name)
    plan = read(root/'plan.json')
    if plan['budget_flops'] != CAP or plan['augmentation'] != 'none' or plan['threshold'] != THRESHOLD:
        raise ValueError('Frozen search policy changed')
    splits = read(root/'splits.json')
    all_val = set()
    for split in splits:
        tr, va = set(split['train']), set(split['val'])
        if len(tr)!=19 or len(va)!=96 or tr&va or all_val&va:
            raise ValueError('Invalid three-fold split')
        all_val |= va
    return plan


def protocol(root, t):
    folder = root/'derived_protocols'/t['protocol']
    folder.mkdir(parents=True, exist_ok=True)
    source = root/'templates'/t['model']
    p = deepcopy(read(source/'protocol.json'))
    descriptor = dict(dataset_root=str(root/'data'), manifest=str(root/'data/fold_0_manifest.json'))
    write(folder/'dataset.json', descriptor)
    for src, name in [(root/'splits.json','splits_10patients.json'),
                      (source/'nnunet_architecture.json','nnunet_architecture.json')]:
        io.copy_verified(src, folder/name)
    p.update(data=str(folder/'dataset.json'), pkl_sha256=digest(folder/'dataset.json'),
             splits_sha256=digest(folder/'splits_10patients.json'), folds=[0,1,2], seeds=[t['seed']],
             baseline_run_id=t['id'])
    write(folder/'protocol.json', p)
    return folder, p


def validate_output(root, out, t):
    """Check scientific results before either seed selection or checkpoint retirement."""
    done, config = read(out/'completion.json'), read(out/'config.json')
    p = read(out/'protocol.json')
    schedule = p['models'][p['trainer_alias']]['objectives']['dice']
    if (done['seed']!=t['seed'] or done['fold']!=t['fold'] or config['model_identity']!=t['model']
            or done['updates']!=schedule['max_steps'] or done['discarded_attempts']
            or done['consumed_training_flops']!=schedule['consumed_training_flops']
            or done['budget_flops']!=CAP or config['smoke_test']):
        raise ValueError('Incomplete or mismatched production result')
    validation = read(out/'validation_latest.json')
    split = read(root/'splits.json')[t['fold']]
    for doc, names in [(validation,split['val']), (read(out/'training_final.json'),split['train'])]:
        cases = doc['cases']
        if len(cases)!=len(names) or {r['case_name'] for r in cases}!=set(names):
            raise ValueError('Missing or duplicated evaluation volumes')
        scores=[]
        for r in cases:
            matrix=r['confusion']
            if len(matrix)!=2 or any(len(a)!=2 for a in matrix) or any(type(v)!=int or v<0 for a in matrix for v in a):
                raise ValueError('Invalid voxel counts')
            if sum(map(sum,matrix)) != 96**3:
                raise ValueError('Incomplete volume evaluation')
            tp=matrix[1][1]; fn=matrix[1][0]; fp=matrix[0][1]
            score=2*tp/(2*tp+fn+fp)
            if abs(score-r['macro_hard_dice'])>1e-10:
                raise ValueError('Dice does not reproduce from counts')
            scores.append(score)
        if abs(sum(scores)/len(scores)-doc['summary']['macro_hard_dice'])>1e-10:
            raise ValueError('Mean Dice does not reproduce')
    return dict(**t, final_dice=validation['summary']['macro_hard_dice'],
                best_dice=done['best_validation_hard_dice'], training_dice=done['final_training_hard_dice'],
                updates=done['updates'], training_flops=done['consumed_training_flops'],
                initialization_sha256=read(out/'initialization.json')['model_state_sha256'])


def result(root, durable, t):
    if not io.completed(durable, t):
        return None
    return validate_output(root,durable/'runs'/t['id'],t)


def load_rows(root, durable):
    rows=[]
    for path in sorted((durable/'runs').glob('*/archive_manifest.json')):
        t=read(path)['task']
        row=result(root,durable,t)
        if row: rows.append(row)
    return rows


def publish(durable, rows, status=None):
    value=summary(rows)
    if status and value['status']!='complete': value['status']=status
    write(durable/'search_status.json',value)
    selected=[r for r in rows if r['seed'] in value['qualified_seeds']]
    for name, data in [('all_attempted_results.csv',rows),('qualified_baselines.csv',selected)]:
        if data:
            tmp=durable/(name+'.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            tmp.replace(durable/name)
    marker=durable/'COMPLETE.json'
    if value['status']=='complete':
        if len(selected)!=36: raise ValueError('Completion requires 36 verified passing cells')
        write(marker,dict(status='complete',qualified_seeds=value['qualified_seeds'],cells=36,
                          minimum_dice=min(r['final_dice'] for r in selected),
                          selected_results_sha256=digest(durable/'qualified_baselines.csv')))
    elif marker.exists():
        raise ValueError('Completion marker conflicts with verified results')
    print(json.dumps({k:v for k,v in value.items() if k not in ('runs','decisions')}),flush=True)
    return value


def preflight(root,durable):
    verify(root); value=io.binding(root,durable)
    expected=dict(status='passed',**value)
    if (durable/'preflight.json').exists() and read(durable/'preflight.json')==expected:
        print('Reusing preflight for this exact bundle and GPU environment.');return
    for model in MODELS:
        t=task(0,0,model);folder,_=protocol(root,t)
        out=root/'preflight'/model;out.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,str(root/'worker.py'),'--protocol-dir',str(folder),
                        '--output',str(out),'--seed','0','--fold','0','--preflight'],
                       env=io.child_environment(root),check=True)
        audit=read(out/'gpu_preflight.json')
        if audit['status']!='passed':raise ValueError('GPU preflight failed')
        io.copy_verified(out/'gpu_preflight.json',durable/'preflight'/f'{model}.json')
    write(durable/'preflight.json',expected)


def run(root,durable,hours,max_runs):
    verify(root);value=io.binding(root,durable)
    if read(durable/'preflight.json')!=dict(status='passed',**value):
        raise ValueError('Run required preflight first')
    deadline=time.time()+hours*3600
    work=root/'work';work.mkdir(exist_ok=True)
    with (work/'search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=load_rows(root,durable);seed=0;executed=0
        try:
            while True:
                state=publish(durable,rows)
                if state['status']=='complete':return
                d=decide(seed,rows)
                if d['status'] in ('qualified','rejected'):
                    print(f"SEED {seed}: {d['status']}",flush=True);seed+=1;continue
                if executed>=max_runs or time.time()>deadline-120 or (durable/'STOP').exists():
                    publish(durable,rows,'paused');return
                t=d['next_task'];folder,p=protocol(root,t)
                output=work/t['id'];recovery=durable/'recovery'/t['id']
                if not output.exists() or not any(output.iterdir()):io.restore_recovery(output,recovery)
                output.mkdir(exist_ok=True)
                io.copy_verified(folder/'protocol.json',output/'protocol.json')
                io.copy_verified(folder/'splits_10patients.json',output/'splits.json')
                if shutil.disk_usage(root).free<12*1024**3:raise RuntimeError('Need 12 GiB runtime SSD headroom')
                io.check_drive_headroom(durable)
                if not (output/'completion.json').exists():
                    command=[sys.executable,'-u',str(root/'worker.py'),'--protocol-dir',str(folder),
                             '--output',str(output),'--seed',str(seed),'--fold',str(t['fold'])]
                    env=io.child_environment(root)
                    env.update(HIPPO_COLAB_RECOVERY=str(recovery),HIPPO_DEADLINE=str(deadline-60))
                    print('RUN',t['id'],flush=True)
                    with (output/'train.log').open('a') as log, (output/'train.log').open() as progress:
                        progress.seek(0,2)
                        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                        sent=False
                        try:
                            while proc.poll() is None:
                                update=progress.read()
                                if update:print(update,end='',flush=True)
                                if (durable/'STOP').exists() and not sent:
                                    proc.send_signal(signal.SIGUSR1);sent=True
                                time.sleep(2)
                            update=progress.read()
                            if update:print(update,end='',flush=True)
                        except KeyboardInterrupt:
                            proc.send_signal(signal.SIGUSR1);proc.wait()
                            publish(durable,rows,'paused');return
                    if proc.returncode==75:publish(durable,rows,'paused');return
                    if proc.returncode:
                        io.copy_verified(output/'train.log',durable/'failures'/t['id']/'train.log')
                        raise RuntimeError('Technical failure; seed not rejected. '+(output/'train.log').read_text()[-3000:])
                candidate=validate_output(root,output,t)
                previous=[r for r in rows if r['seed']==seed and r['model']==t['model']]
                if any(r['initialization_sha256']!=candidate['initialization_sha256'] for r in previous):
                    raise ValueError('Initialization changed across folds; recovery preserved')
                io.archive_completed(root,output,durable,recovery,t)
                row=result(root,durable,t)
                if row is None:raise ValueError('Archive failed completion verification')
                rows.append(row);executed+=1
                print(f"RESULT {t['id']}: {100*row['final_dice']:.3f}%",flush=True)
        except Exception as exc:
            write(durable/'search_error.json',dict(error=str(exc),time=time.time()))
            publish(durable,rows,'technical_error');raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['verify','preflight','run','report'])
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--durable',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=10)
    parser.add_argument('--max-runs',type=int,default=10000)
    args=parser.parse_args()
    if args.hours<=0 or args.max_runs<1:parser.error('Positive session limits required')
    root=io.activate(args.bundle);durable=args.durable.resolve();durable.mkdir(parents=True,exist_ok=True)
    if args.command=='verify':verify(root);print('Code/data/splits verified')
    elif args.command=='preflight':preflight(root,durable)
    elif args.command=='report':verify(root);publish(durable,load_rows(root,durable))
    else:run(root,durable,args.hours,args.max_runs)


if __name__=='__main__':main()
