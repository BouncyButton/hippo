"""CE2 baseline and conditional matched BANDS, each <= original cap including calibration."""
import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time

import runtime as engine
io=engine.io
CAP=engine.CAP
TOTAL_CAP=CAP*2
ARMS=('ce2','ce2_bands')
LRS={arm:1e-4 for arm in ARMS}
INITIALIZATION='1215193efc51fc67a9da9cb1ef0bca0e6e8019d4af8146d9efcb85b94bb051f3'


def task(arm):
    if arm not in ARMS:raise ValueError('Unknown rescue arm')
    return dict(id='GeoFWI3D_unetr_'+arm+'_f0s0',kind='conventional',
        dataset='GeoFWI3D',model='unetr_vit_b16',fold=0,seed=0,arm='dice_bands' if arm=='ce2_bands' else 'dice',protocol=arm,augmentation='xy_d4_v1')


def schedule(root,arm):
    p=io.read(root/'templates/unetr_vit_b16/protocol.json')
    return p['models']['unet']['objectives'][task(arm)['arm']]


def training_cost(n,s):
    prefix=s['prefix_updates']
    return min(n,prefix)*s['baseline_flops_per_update']+max(0,n-prefix)*s['flops_per_update']


def total_cost(n,s):
    return training_cost(n,s)+(s['calibration_flops'] if n>s['prefix_updates'] else 0)


def fraction_step(root,arm,i):
    s=schedule(root,arm)
    return max(n for n in range(1,s['max_steps']+1) if total_cost(n,s)<=CAP*i//10)


def settings(root,arm='ce2'):
    s=schedule(root,arm)
    return s['flops_per_update'],fraction_step(root,arm,5),s['max_steps']


def verify(root):
    p=engine.verify(root)
    if p['maximum_optimization_flops_total']!=TOTAL_CAP:raise ValueError('Total cap differs')
    profile=io.read(root/'rescue_profile.json')
    if profile['recipe_sha256']!=io.digest(root/'rescue_objective.py'):raise ValueError('Profile recipe differs')
    for arm in ARMS:
        s=schedule(root,arm)
        if total_cost(s['max_steps'],s)>CAP or total_cost(s['max_steps']+1,s)<=CAP:
            raise ValueError('Invalid capped schedule')
    return p


def protocol(root,arm):
    folder,p=engine.protocol(root,task(arm))
    p.update(initial_lrs={'unet':LRS[arm]},rescue_arm=arm,folds=[0],seeds=[0],objective=task(arm)['arm'],
        rescue_validation_steps=sorted({fraction_step(root,arm,i) for i in range(1,11)}|{schedule(root,arm)['max_steps']}))
    io.write(folder/'protocol.json',p)
    return folder,p


def valid_evaluation(doc,names):
    cases=doc['cases']
    if len(cases)!=len(names) or {r['case_name'] for r in cases}!=set(names):
        raise ValueError('Evaluation case set differs')
    scores=[]
    for r in cases:
        m=r['confusion']
        if len(m)!=2 or any(len(row)!=2 for row in m) or any(type(v)!=int or v<0 for row in m for v in row):
            raise ValueError('Invalid confusion counts')
        if sum(map(sum,m))!=96**3:raise ValueError('Incomplete evaluation volume')
        tp=m[1][1];fp=m[0][1];fn=m[1][0]
        score=2*tp/(2*tp+fp+fn)
        if abs(score-r['macro_hard_dice'])>1e-10:raise ValueError('Invalid case Dice')
        scores.append(score)
    if abs(sum(scores)/len(scores)-doc['summary']['macro_hard_dice'])>1e-10:
        raise ValueError('Invalid mean Dice')


def check_recipe(out,arm):
    recipe=io.read(out/'runtime_recipe.json');config=io.read(out/'config.json')
    if (recipe['dropout_rate']!=.1 or not recipe['squared_pred'] or recipe['dice_include_background']
        or recipe['weight_ce']!=2 or recipe['weight_dice']!=1 or recipe['augmentation']!='xy_d4_v1' or config['augmentation']!='xy_d4_v1'
        or config['initial_lr']!=LRS[arm] or config['fold']!=0 or config['seed']!=0
        or config['smoke_test'] or config['lr_horizon_updates']!=477):
        raise ValueError('Rescue recipe differs')
    if io.read(out/'initialization.json')['model_state_sha256']!=INITIALIZATION:
        raise ValueError('Initialization differs')


def screening_score(rows,steps):
    by_step={int(r['step']):r for r in rows}
    if len(by_step)!=len(rows):raise ValueError('Duplicate validation checkpoint')
    scores=[float(by_step[s]['macro_hard_dice']) for s in steps]
    if any(not math.isfinite(x) or not 0<=x<=1 for x in scores):raise ValueError('Invalid score')
    return sum(scores)/2,steps,scores


def check_screen(root,out,arm):
    cost,half,full=settings(root,arm);stage=io.read(out/'stage_complete.json');s=schedule(root,arm)
    ledger=io.read(out/'budget_ledger.json');check_recipe(out,arm)
    if (stage['updates']!=half or stage['charged_attempts']!=half
        or stage['training_flops']!=training_cost(half,s) or ledger['training_flops']!=training_cost(half,s)
        or ledger['charged_attempts']!=half or ledger['calibration_flops']!=s['calibration_flops']):
        raise ValueError('Screening budget differs')
    splits=io.read(root/'splits.json')[0]
    valid_evaluation(io.read(out/'validation_latest.json'),splits['val'])
    valid_evaluation(io.read(out/'validation_best.json'),splits['val'])
    valid_evaluation(io.read(out/'training_stage_final.json'),splits['train'])
    with (out/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    score,steps,scores=screening_score(rows,[fraction_step(root,arm,i) for i in (4,5)])
    return dict(arm=arm,status='screened',updates=half,training_flops=training_cost(half,s),
        calibration_flops=ledger['calibration_flops'],total_flops=total_cost(half,s),
        selection_score=score,selection_steps=steps,selection_scores=scores,
        half_budget_dice=io.read(out/'validation_latest.json')['summary']['macro_hard_dice'])


def read_screen(root,durable,arm):
    folder=durable/'screening'/task(arm)['id'];manifest=folder/'screen_manifest.json'
    if not manifest.exists():return None
    for name,sha in io.read(manifest)['files'].items():
        if io.digest(folder/name)!=sha:raise ValueError('Screen archive changed: '+name)
    return check_screen(root,folder,arm)


def archive_screen(root,out,durable,recovery,arm):
    row=check_screen(root,out,arm)
    receipt=io.read(recovery/'latest.json')
    if receipt['charged_attempts']!=row['updates'] or io.digest(recovery/receipt['file'])!=receipt['sha256']:
        raise ValueError('Half-budget checkpoint was not durably saved')
    if io.digest(out/'checkpoint_latest.pt')!=io.read(out/'stage_complete.json')['checkpoint_sha256']:
        raise ValueError('Half-budget model changed')
    target=durable/'screening'/task(arm)['id'];files={}
    for f in out.iterdir():
        if f.is_file() and f.suffix in ('.json','.csv','.log'):
            io.copy_verified(f,target/f.name);files[f.name]=io.digest(f)
    io.write(target/'screen_manifest.json',dict(files=files,task=task(arm),
        bundle_sha256=io.digest(root/'bundle_manifest.json'),recovery=receipt))
    if read_screen(root,durable,arm)!=row:raise ValueError('Screen archive readback differs')
    return row


def gate(row):
    score=row['selection_score']
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid gate score')
    return dict(winner=row['arm'] if score>=.46 else None, threshold=.46,
        score=score, decision='continue' if score>=.46 else 'prune',
        rule='Mean validation Dice at 40% and 50% of the capped allowance must be >= 0.46')


def prune(root,durable,arm):
    target=durable/'screening'/task(arm)['id']
    if (target/'pruned.json').exists():return
    read_screen(root,durable,arm)
    out=root/'work'/task(arm)['id'];recovery=durable/'recovery'/task(arm)['id']
    manifest=io.read(target/'screen_manifest.json');r=manifest['recovery'];deleted=[]
    # The chosen arm still depends on its checkpoint. Only retire the declared loser.
    if io.read(durable/(arm+'_selection.json'))['winner']==arm:raise ValueError('Cannot prune winner')
    if out.exists():
        stage=io.read(target/'stage_complete.json')
        for name,key in [('checkpoint_latest.pt','checkpoint_sha256'),('checkpoint_best.pt','best_checkpoint_sha256')]:
            f=out/name
            if f.exists():
                if io.digest(f)!=stage[key]:raise ValueError('Pruned checkpoint changed')
                f.unlink();deleted.append(str(f))
    f=recovery/io.safe_member(r['file'])
    if f.exists():
        if io.digest(f)!=r['sha256']:raise ValueError('Pruned recovery changed')
        f.unlink();deleted.append(str(f))
    for name in ('latest.json','charged_ledger.json'):(recovery/name).unlink(missing_ok=True)
    io.write(target/'pruned.json',dict(status='pruned',reason='Mean validation Dice at 40% and 50% budget below 0.46',
        deleted=deleted,archive_verified=True,full_budget_result=False,time=time.time()))


def final_result(root,durable,arm):
    t=task(arm)
    if not io.completed(durable,t):return None
    folder=durable/'runs'/t['id'];check_recipe(folder,arm)
    return engine.validate_output(root,folder,t)


def qualifies(row):
    return bool(row and math.isfinite(row['final_dice']) and row['final_dice']>.5)


def comparison(root,durable,baseline,bands):
    if not (baseline and bands):return None
    documents=[io.read(durable/'runs'/task(arm)['id']/'validation_latest.json') for arm in ARMS]
    case_maps=[{r['case_name']:r for r in doc['cases']} for doc in documents]
    rows=[]
    for name in sorted(case_maps[0]):
        a,b=[m[name] for m in case_maps]
        ca,cb=a['confusion'],b['confusion']
        rows.append(dict(case_name=name,ground_truth_voxels=sum(ca[1]),
            baseline_dice=a['macro_hard_dice'],bands_dice=b['macro_hard_dice'],
            delta=b['macro_hard_dice']-a['macro_hard_dice'],
            baseline_fp=ca[0][1],bands_fp=cb[0][1],baseline_fn=ca[1][0],bands_fn=cb[1][0]))
    ordered=sorted(rows,key=lambda r:(r['ground_truth_voxels'],r['case_name']))
    quartiles=[dict(quartile=i+1,cases=len(ordered[i*24:(i+1)*24]),
        mean_delta=sum(r['delta'] for r in ordered[i*24:(i+1)*24])/24) for i in range(4)]
    report=dict(final_dice_delta_percentage_points=100*(bands['final_dice']-baseline['final_dice']),
        cases=rows,salt_volume_quartiles=quartiles,
        note='One paired development result, not a three-fold/three-seed consistency claim')
    io.write(durable/'paired_comparison.json',report)
    return report['final_dice_delta_percentage_points']


def publish(root,durable,status=None):
    screens=[r for arm in ARMS if (r:=read_screen(root,durable,arm)) is not None]
    finals={arm:final_result(root,durable,arm) for arm in ARMS}
    pruned={arm:(durable/'screening'/task(arm)['id']/'pruned.json').exists() for arm in ARMS}
    spent={arm:0 for arm in ARMS}
    for row in screens:spent[row['arm']]=row['total_flops']
    for arm in ARMS:
        for f in [root/'work'/task(arm)['id']/'budget_ledger.json',
                  durable/'recovery'/task(arm)['id']/'charged_ledger.json']:
            if f.exists():
                ledger=io.read(f);spent[arm]=max(spent[arm],ledger['training_flops']+ledger['calibration_flops'])
        if finals[arm]:spent[arm]=finals[arm]['total_flops']
        if spent[arm]>CAP:raise ValueError('Per-arm cap exceeded')
    passed=qualifies(finals['ce2'])
    if not passed and spent['ce2_bands']:raise ValueError('BANDS started without qualifying baseline')
    terminal=(pruned['ce2'] or bool(finals['ce2'] and not passed)
              or bool(passed and (finals['ce2_bands'] or pruned['ce2_bands'])))
    value=dict(status=('pruned' if pruned['ce2'] else 'complete') if terminal else (status or 'in_progress'),
        scope='One CE2 baseline and conditional matched BANDS; not three-seed baseline completion',
        screening=screens,final=finals['ce2'],bands=finals['ce2_bands'],pruned=pruned,
        total_flops_by_arm=spent,total_flops=sum(spent.values()),cap_per_arm=CAP,maximum_total_flops=TOTAL_CAP,
        threshold_passed=passed,bands_authorized_to_launch=passed,
        bands_status='complete' if finals['ce2_bands'] else ('pruned' if pruned['ce2_bands'] else ('pending_or_running' if passed else 'not_triggered')),
        paired_delta_percentage_points=comparison(root,durable,finals['ce2'],finals['ce2_bands']))
    io.write(durable/'ce2_bands_status.json',value)
    if terminal:io.write(durable/'EXPERIMENT_FINISHED.json',value)
    print(json.dumps(value,indent=2),flush=True)
    return value


def execute(root,durable,arm,limit,deadline):
    t=task(arm);folder,p=protocol(root,arm)
    out=root/'work'/t['id'];recovery=durable/'recovery'/t['id']
    if time.time()>deadline-120 or (durable/'STOP').exists():return False
    if not out.exists() or not (out/'checkpoint_latest.pt').exists():
        io.restore_recovery(out,recovery)
    out.mkdir(parents=True,exist_ok=True)
    io.copy_verified(folder/'protocol.json',out/'protocol.json')
    io.copy_verified(folder/'splits_10patients.json',out/'splits.json')
    if shutil.disk_usage(root).free<16*1024**3:raise RuntimeError('Need 16 GiB runtime disk headroom')
    io.check_drive_headroom(durable)
    env=io.child_environment(root)
    env.update(HIPPO_COLAB_RECOVERY=str(recovery),HIPPO_DEADLINE=str(deadline-60),HIPPO_STAGE_LIMIT=str(limit))
    cmd=[sys.executable,'-u',str(root/'worker.py'),'--protocol-dir',str(folder),
         '--output',str(out),'--seed','0','--fold','0']
    if not (out/'completion.json').exists():
        print('RUN',arm,'through update',limit,flush=True)
        with (out/'train.log').open('a') as log,(out/'train.log').open() as progress:
            progress.seek(0,2)
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            sent=False
            try:
                while proc.poll() is None:
                    update=progress.read()
                    if update:print(update,end='',flush=True)
                    if (durable/'STOP').exists() and not sent:
                        proc.send_signal(signal.SIGUSR1);sent=True
                    time.sleep(2)
                print(progress.read(),end='',flush=True)
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGUSR1);proc.wait()
                return False
        if proc.returncode==75:return False
        if proc.returncode not in (0,76):
            io.copy_verified(out/'train.log',durable/'failures'/t['id']/'train.log')
            raise RuntimeError('Training failed; recovery preserved. '+(out/'train.log').read_text()[-2500:])
    if limit==settings(root,arm)[1]:archive_screen(root,out,durable,recovery,arm)
    else:
        check_recipe(out,arm);engine.validate_output(root,out,t)
        io.archive_completed(root,out,durable,recovery,t)
    return True


def preflight(root,durable,arm='ce2'):
    value=io.binding(root,durable)
    if value['environment']!=io.read(root/'control_environment.json'):
        raise ValueError('Match archived A100-80GB environment before paired training')
    expected=dict(status='passed',**value)
    if (durable/(arm+'_preflight.json')).exists() and io.read(durable/(arm+'_preflight.json'))==expected:return
    # Both arms share the model/loss/optimizer implementation; test both LRs explicitly.
    for arm in (arm,):
        folder,_=protocol(root,arm);out=root/'preflight'/arm
        subprocess.run([sys.executable,str(root/'worker.py'),'--protocol-dir',str(folder),
            '--output',str(out),'--seed','0','--fold','0','--preflight'],env=io.child_environment(root),check=True)
        if io.read(out/'gpu_preflight.json')['status']!='passed':raise ValueError('GPU preflight failed')
        io.copy_verified(out/'gpu_preflight.json',durable/'preflight'/f'{arm}.json')
    io.write(durable/(arm+'_preflight.json'),expected)


def run(root,durable,hours):
    value=io.binding(root,durable)
    if io.read(durable/'ce2_preflight.json')!=dict(status='passed',**value):raise ValueError('Run preflight first')
    work=root/'work';work.mkdir(exist_ok=True)
    with (work/'augmentation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.time()+hours*3600
        try:
            if publish(root,durable)['status'] in ('complete','pruned'):return
            for arm in ARMS:
                if arm=='ce2_bands':
                    baseline=final_result(root,durable,'ce2')
                    if not qualifies(baseline):publish(root,durable);return
                    if time.time()>deadline-300:publish(root,durable,'paused');return
                    preflight(root,durable,arm)
                cost,half,full=settings(root,arm)
                if not io.completed(durable,task(arm)):
                    if read_screen(root,durable,arm) is None:
                        if not execute(root,durable,arm,half,deadline):publish(root,durable,'paused');return
                    decision=gate(read_screen(root,durable,arm));path=durable/(arm+'_selection.json')
                    if path.exists() and io.read(path)!=decision:raise ValueError('Gate changed')
                    io.write(path,decision);print('HALF-BUDGET GATE',arm,decision,flush=True)
                    if decision['decision']=='prune':
                        prune(root,durable,arm);publish(root,durable);return
                    if not execute(root,durable,arm,full,deadline):publish(root,durable,'paused');return
                publish(root,durable)
        except Exception as exc:
            io.write(durable/'ce2_bands_error.json',dict(error=str(exc),time=time.time()))
            publish(root,durable,'technical_error');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','preflight','run','report'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--durable',type=Path,required=True)
    p.add_argument('--hours',type=float,default=2);args=p.parse_args()
    if not math.isfinite(args.hours) or args.hours<=0:p.error('Positive finite hours required')
    root=io.activate(args.bundle);durable=args.durable.resolve();durable.mkdir(parents=True,exist_ok=True)
    verify(root)
    if args.command=='verify':print('Rescue code/data/profile verified',settings(root))
    elif args.command=='preflight':preflight(root,durable)
    elif args.command=='run':run(root,durable,args.hours)
    else:publish(root,durable)


if __name__=='__main__':main()
