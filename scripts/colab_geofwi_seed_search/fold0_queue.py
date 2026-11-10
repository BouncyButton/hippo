"""Five baseline jobs on fold 0, followed by a strict twelve-cell qualification audit."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

CAP = 182178126543160
MODELS = ('nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def archive_row(folder, model, seed, splits):
    manifest = read(folder/'archive_manifest.json')
    task = manifest['task']
    assert (task['model'], task['fold'], task['seed'], task['arm']) == (model, 0, seed, 'dice')
    for name, info in manifest['metadata'].items():
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        f = folder/p
        assert sha(f) == info['sha256'] and f.stat().st_size == info['bytes'], name
    done, config = read(folder/'completion.json'), read(folder/'config.json')
    protocol = read(folder/'protocol.json')
    schedule = protocol['models'][protocol['trainer_alias']]['objectives']['dice']
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert done['budget_flops'] == CAP and done['consumed_total_flops'] <= CAP
    assert done['updates'] == schedule['max_steps']
    assert done['consumed_training_flops'] == schedule['consumed_training_flops']
    assert (config['model_identity'], config['seed'], config['fold']) == (model, seed, 0)
    assert not config['smoke_test']
    assert read(folder/'splits.json') == splits
    expected_lr = {'nnunet':.01, 'swinunetr':1e-4, 'unetr_vit_b16':6e-4, 'basic_unet16':3e-4}[model]
    assert config['initial_lr'] == expected_lr
    recipe = read(folder/'runtime_recipe.json')
    assert recipe['weight_ce'] == (1 if model in ('nnunet', 'unetr_vit_b16') else 0)
    assert config['augmentation'] == ('xy_d4_v1' if model == 'unetr_vit_b16' else 'none')
    for filename, names in [('validation_latest.json',splits[0]['val']), ('training_final.json',splits[0]['train'])]:
        doc = read(folder/filename)
        assert len(doc['cases']) == len(names) and {r['case_name'] for r in doc['cases']} == set(names)
        scores=[]
        for row in doc['cases']:
            c=row['confusion']
            assert len(c)==2 and all(len(x)==2 for x in c)
            assert all(type(v)==int and v>=0 for x in c for v in x) and sum(map(sum,c))==96**3
            score=2*c[1][1]/(2*c[1][1]+c[0][1]+c[1][0])
            assert abs(score-row['macro_hard_dice'])<1e-10
            scores.append(score)
        assert abs(sum(scores)/len(scores)-doc['summary']['macro_hard_dice'])<1e-10
    return dict(model=model, seed=seed, fold=0, final_dice=read(folder/'validation_latest.json')['summary']['macro_hard_dice'],
        updates=done['updates'], total_flops=done['consumed_total_flops'], source=str(folder),
        archive_manifest_sha256=sha(folder/'archive_manifest.json'))


def qualification(rows):
    expected={(m,s) for m in MODELS for s in range(3)}
    keys=[(r['model'],r['seed']) for r in rows]
    if len(set(keys))!=len(keys) or set(keys)-expected:
        raise ValueError('Duplicate or unexpected baseline')
    if any(r['fold']!=0 or not math.isfinite(r['final_dice']) or not 0<=r['final_dice']<=1 for r in rows):
        raise ValueError('Invalid result')
    missing=sorted(expected-set(keys))
    failed=[r for r in rows if r['final_dice']<=.5]
    return dict(qualified=not missing and not failed, missing=missing, failed=failed,
        threshold=.5, comparison='strictly_greater', baseline_cells=len(rows), required_cells=12,
        scope='Fold 0, seeds 0/1/2 only; not completion of the three-fold study',
        interpretation='UNETR uses the selected augmented LR6 recipe; other models retain established recipes. Validation-selected development cohort.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--durable',type=Path,required=True)
    ap.add_argument('--previous-results',type=Path,required=True)
    ap.add_argument('--hours',type=float,default=6)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    if not math.isfinite(args.hours) or not 0<args.hours<=6:ap.error('Hours must be in (0,6]')
    root,durable=args.bundle.resolve(),args.durable.resolve()
    for name,expected in read(root/'manifest.json').items():
        assert sha(root/name)==expected, name
    splits=read(root/'unetr/splits.json')
    assert splits==read(root/'basic/splits.json')
    refs=[archive_row(args.previous_results/'runs'/f'GeoFWI3D_{m}_baseline_f0s{s}',m,s,splits)
        for m in ('nnunet','swinunetr') for s in range(3)]
    refs.append(archive_row(root/'reference_unetr_lr6_seed0','unetr_vit_b16',0,splits))
    durable.mkdir(parents=True,exist_ok=True)
    write(durable/'verified_references.json',refs)
    deadline=time.time()+args.hours*3600
    with (durable/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=list(refs)
        for child,model in [('unetr','unetr_vit_b16'),('basic','basic_unet16')]:
            command=[sys.executable,'-u',str(root/child/'fold0_child.py'),'--bundle',str(root/child),
                '--durable',str(durable/child),'--model',model,'--deadline',str(deadline)]
            if args.verify_only:command.append('--verify-only')
            write(durable/'fold0_status.json',dict(status='verifying' if args.verify_only else 'running',active_model=model,
                rows=rows,**qualification(rows)))
            print('QUEUE',model,flush=True)
            proc=subprocess.Popen(command)
            try:code=proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(2);proc.wait();code=75
            if code:
                write(durable/'fold0_status.json',dict(status='paused' if code==75 else 'technical_error',
                    active_model=model,rows=rows,**qualification(rows)))
                raise SystemExit(code)
            if not args.verify_only:
                for row in read(durable/child/'results.json'):
                    rows.append(archive_row(durable/child/'runs'/row['id'],model,row['seed'],splits))
        if args.verify_only:
            print('All references, capped schedules and requested tasks verified',flush=True)
            return
        result=dict(status='complete' if qualification(rows)['qualified'] else 'finished_not_qualified',
            rows=rows,**qualification(rows))
        write(durable/'fold0_status.json',result)
        write(durable/'FIVE_RUNS_FINISHED.json',result)
        if result['qualified']:write(durable/'FOLD0_QUALIFIED.json',result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
