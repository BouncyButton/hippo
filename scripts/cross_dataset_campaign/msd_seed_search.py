"""Bounded, auditable MSD seed screening and all-fold qualification."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import CAP, MODELS, read, write, digest, install_signals

ROOT = Path(__file__).resolve().parents[2]
NAME = 'msd_seed_search_10_41_20261018'
MODULE = 'scripts.cross_dataset_campaign.msd_seed_search'
PARENT = 'msd_four_models_seeds789_baselines_20261018'
PARENT_SHA = 'e5f30b022882dffcda908dc775eac3013e10e5a84f8e432d7c539d99990942b0'
SEEDS = tuple(range(10, 42))
TARGET = .5
SCREEN = {'unetr_vit_b16': .28, 'swinunetr': .39}
CLI = 'source/scripts/cross_dataset_campaign/capped_core.py'
OLD = 'parser.add_argument("--seed", type=int, choices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), required=True)'
NEW = 'parser.add_argument("--seed", type=int, choices=tuple(range(42)), required=True)'
BASE_SUBMIT = controller.submit


def task(seed, model, fold=0, phase='full'):
    return dict(id=f'MSD_{model}_dice_f{fold}s{seed}_{phase}', kind='conventional',
                dataset='MSD', model=model, seed=seed, fold=fold, arm='dice', phase=phase,
                protocol=f'MSD_{model}_f{fold}s7')


def tasks():
    return [task(s, m, f) for s in SEEDS for f in range(3) for m in MODELS] + [
        task(s, m, phase='screen') for s in SEEDS for m in SCREEN]


def recipe(original, item, initialization):
    p = deepcopy(original)
    p.update(name=NAME, seeds=[item['seed']], folds=[item['fold']],
        baseline_run_id=item['id'], baseline_reference=dict(seed=item['seed'],
            initialization_sha256=initialization, policy='Independently fingerprint initialization before each run; compare all folds and screen/full identities'),
        seed_search_phase=item['phase'], requested_arm_order=['dice'], constraints=False)
    if item['phase'] == 'screen':
        budget = CAP//2
        p.update(budget_flops=budget, budget_gflops=budget/1e9)
        schedule = p['models'][p['trainer_alias']]['objectives']['dice']
        steps = budget//schedule['flops_per_update']; consumed = steps*schedule['flops_per_update']
        schedule.update(max_steps=steps, consumed_training_flops=consumed,
            consumed_flops=consumed, consumed_total_flops=consumed,
            unused_flops=budget-consumed, patient_pass_epochs=steps/10)
    return p


def prepare():
    stage = ROOT/'experiments'/NAME
    if stage.exists():
        verify(stage); return stage
    parent = ROOT/'reports'/PARENT/'frozen'
    if digest(parent/'manifest.json') != PARENT_SHA:
        raise ValueError('Baseline provenance changed')
    for name, sha in read(parent/'manifest.json').items():
        if digest(parent/name) != sha: raise ValueError('Parent changed: '+name)
    stage.mkdir(parents=True)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    shutil.copytree(parent/'source', stage/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = (stage/CLI).read_text()
    if source.count(OLD) != 1: raise ValueError('Unexpected seed CLI')
    (stage/CLI).write_text(source.replace(OLD, NEW))
    shutil.copy2(Path(__file__), stage/'source/scripts/cross_dataset_campaign/msd_seed_search.py')
    for f in range(3):
        for m in MODELS:
            n=f'MSD_{m}_f{f}s7'
            shutil.copytree(parent/'protocols'/n, stage/'protocols'/n)
    plan = deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME, tasks=tasks(), num_runs=len(tasks()), seeds=list(SEEDS),
        acceptance=dict(minimum_final_dice=TARGET, required_models=list(MODELS), required_folds=[0,1,2]),
        screening=dict(fold=0, fraction=.5, cutoffs=SCREEN,
            evidence='Historical half-budget minimum among final>=50% runs: UNETR33.085%, Swin44.455%. Cutoffs28%/39% leave about5pp margin; early rejection can still miss late bloomers.',
            full_run_policy='Independent full-budget rerun after screening; screening FLOPs count separately in search total. Preserve original final-run trajectory.'),
        max_search_training_flops=32*13*CAP,
        authorization='User authorized seeds10–41, maximum24h, stop at first seed with>=50% final Dice for every model on every fold. Baselines only, original no-augmentation FLOP-capped recipe; two GPU workers.',
        comparison_policy='Exploratory selection using existing validation folds, not an unbiased held-out test. All screened, rejected and full runs retained.',
        exploration_followup='No automatic successor or seeds outside10–41; explicitly report no winner when exhausted or time-limited.')
    plan['storage']['archive']=str(ROOT/'reports'/NAME)
    plan['parallelism'].update(datasets=['MSD'], assignment='sequential_candidate_stages_two_workers')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False,disabled_tasks=[]))
    batch=(parent/'run.sbatch').read_text().replace('hippo-msd-seeds789','hippo-msd-seed-search')
    batch=batch.replace('scripts.cross_dataset_campaign.seeds789_msd_baselines_job', MODULE)
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*'))
        if p.is_file() and p.name not in ('manifest.json','control.json')})
    verify(stage);return stage


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:raise ValueError('Frozen search changed: '+name)
    if digest(stage/'parent_manifest.json')!=PARENT_SHA:raise ValueError('Wrong parent')
    for name,sha in read(stage/'parent_manifest.json').items():
        if name.startswith('source/'):
            actual=digest(stage/name)
            if name==CLI:
                source=(stage/name).read_text()
                if source.count(NEW)!=1:raise ValueError('Seed range changed')
                actual=hashlib.sha256(source.replace(NEW,OLD).encode()).hexdigest()
            if actual!=sha:raise ValueError('Scientific source changed: '+name)
        if name.startswith('protocols/MSD_') and 's7/' in name and digest(stage/name)!=sha:
            raise ValueError('Original template changed')
    p=read(stage/'plan.json')
    if (p['tasks']!=tasks() or p['seeds']!=list(SEEDS) or p['acceptance']['minimum_final_dice']!=.5
            or p['screening']['cutoffs']!=SCREEN or p['max_job_hours']!=24
            or p['max_search_training_flops']!=32*13*CAP or p['parallelism']['workers']!=2
            or p['storage']['minimum_free_gib']<20 or 'archive_resume_module' in p):
        raise ValueError('Search scope changed')
    return p


def worker(stage,item):
    """Derive and archive the exact runtime protocol; training remains unchanged."""
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    from scripts.cross_dataset_campaign import seed3_baselines_gpu_worker as gpu
    if item not in tasks():raise ValueError('Unauthorized task')
    output=stage/'runs'/item['id']; output.mkdir(parents=True,exist_ok=True)
    original=stage/'protocols'/item['protocol']; p=read(original/'protocol.json')
    torch.set_num_threads(2);exp.base.seed_everything(item['seed'])
    model=exp.build_model(p,read(original/'nnunet_architecture.json'))
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    del model
    p=recipe(p,item,h.hexdigest())
    folder=output/'protocol_bundle';folder.mkdir(exist_ok=True)
    for path in original.iterdir():
        if path.name!='protocol.json':
            target=folder/path.name
            if target.exists() and digest(target)!=digest(path):raise ValueError('Resume template changed')
            if not target.exists():shutil.copy2(path,target)
    target=folder/'protocol.json'
    if target.exists() and read(target)!=p:raise ValueError('Runtime protocol changed')
    if not target.exists():write(target,p)
    write(output/'search_task.json',dict(task=item,template_sha256=digest(original/'protocol.json'),
        runtime_protocol_sha256=digest(target),initialization_sha256=h.hexdigest(),search_manifest_sha256=digest(stage/'manifest.json')))
    sys.argv=['gpu-worker','--record',str(output/'gpu_worker.json'),'--model',p['trainer_alias'],
        '--objective','dice','--fold',str(item['fold']),'--seed',str(item['seed']),
        '--pkl',p['data'],'--protocol-dir',str(folder),'--output',str(output),'--resume']
    gpu.main()


def worker_command(stage,plan,item,base,lane):
    from scripts.cross_dataset_campaign.seeds789_msd_parallel import step_prefix
    return [*step_prefix(),'env',f'HIPPO_GPU_WORKER={lane}',plan['conventional_python'],'-m',MODULE,
            'worker','--stage',str(stage),'--task',json.dumps(item,separators=(',',':'))]


def score(stage,item):
    folder=stage/'runs'/item['id']
    if read(folder/'cleanup.json')['status']!='complete':raise ValueError('Archive/cleanup incomplete')
    done=read(folder/'completion.json');p=read(folder/'protocol_bundle/protocol.json')
    expected=recipe(read(stage/'protocols'/item['protocol']/'protocol.json'),item,p['baseline_reference']['initialization_sha256'])
    if p!=expected or done['updates']!=p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']:
        raise ValueError('Run does not match planned budget')
    value=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid final Dice')
    return value


def decision(stage,seed):
    """Return the next two-or-fewer runs, rejection, or fully verified winner."""
    def finished(t):
        p=stage/'runs'/t['id']/'cleanup.json'
        return p.exists() and read(p)['status']=='complete'
    stages=[
        [task(seed,m,phase='screen') for m in ('unetr_vit_b16','swinunetr')],
        [task(seed,m) for m in ('unetr_vit_b16','swinunetr')],
        [task(seed,m) for m in ('nnunet','basic_unet16')],
    ]
    stages += [[task(seed,m,f) for m in ('unetr_vit_b16','swinunetr')]
               for f in (1,2)]
    stages += [[task(seed,m,f) for m in ('nnunet','basic_unet16')]
               for f in (1,2)]
    for group in stages:
        for t in group:
            if finished(t):
                threshold=SCREEN[t['model']] if t['phase']=='screen' else TARGET
                value=score(stage,t)
                if value<threshold:return dict(outcome='rejected',task=t['id'],dice=value,threshold=threshold)
        missing=[t for t in group if not finished(t)]
        if missing:return dict(outcome='next',tasks=missing)
    full=[task(seed,m,f) for f in range(3) for m in MODELS]
    # Require identical initialization across all folds and the screen/full pair.
    for m in MODELS:
        relevant=[t for t in tasks() if t['seed']==seed and t['model']==m]
        hashes={read(stage/'runs'/t['id']/'initialization.json')['model_state_sha256'] for t in relevant}
        if len(hashes)!=1:raise ValueError('Initialization differs across matching runs')
    values=[dict(model=t['model'],fold=t['fold'],dice=score(stage,t)) for t in full]
    if len(values)!=12 or min(v['dice'] for v in values)<TARGET:raise ValueError('Invalid winner')
    return dict(outcome='qualified',seed=seed,results=values,minimum_dice=min(v['dice'] for v in values))


def run(stage):
    import fcntl
    import traceback
    from scripts.cross_dataset_campaign import seeds789_msd_parallel as parallel
    plan=verify(stage);install_signals()
    os.environ['HIPPO_CONTROL']=str(stage/'control.json')
    os.environ['HIPPO_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    parallel.worker_command=worker_command
    decisions=[]; winner=None
    def publish(status,workers=None,outcome=None):
        active=[w['task']['id'] for w in (workers or {}).values()]
        controller.summarize(stage,plan,status,active)
        summary=read(stage/'status.json')
        total=sum(read(p)['training_flops'] for p in (stage/'runs').glob('*/budget_ledger.json'))
        summary.update(outcome=outcome,decisions=decisions,winner=winner,
                       search_training_flops=total,search_budget_flops=plan['max_search_training_flops'])
        if total>plan['max_search_training_flops']:raise ValueError('Search compute bound exceeded')
        write(stage/'status.json',summary);write(stage/'search_summary.json',summary)
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            parallel.verify_gpu_probe(stage,plan)
            for seed in SEEDS:
                while True:
                    d=decision(stage,seed)
                    if d['outcome']!='next':
                        decisions.append(dict(seed=seed,**{k:v for k,v in d.items() if k!='seed'}))
                        if d['outcome']=='qualified':
                            winner=d;publish('complete',outcome='qualified_seed');return
                        publish('running');break
                    wave=deepcopy(plan);wave['tasks']=d['tasks']
                    result=parallel.dispatch(stage,wave,None,lambda state,workers:publish('running' if state=='complete' else state,workers))
                    if result!='complete':
                        if result=='continuing_at_time_limit':publish('complete',outcome='time_limit_no_qualified_seed')
                        else:publish(result,outcome='search_interrupted_no_qualified_seed')
                        return
            publish('complete',outcome='candidate_range_exhausted_no_qualified_seed')
        except Exception:
            write(stage/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()))
            publish('failed',outcome='failure_no_qualified_seed');raise


def preflight(stage):
    import importlib.metadata
    import tempfile
    from unittest.mock import patch
    import torch
    from scripts.cross_dataset_campaign import conventional as exp
    from scripts.cross_dataset_campaign.common import ARMS
    plan=verify(stage);torch.set_num_threads(2);exp.trainer.ARMS=ARMS
    if str(torch.__version__)!='2.12.1+cu130' or importlib.metadata.version('monai')!='1.6.0':
        raise ValueError('Original environment changed')
    template=read(stage/'protocols/MSD_swinunetr_f0s7/protocol.json')
    if digest(template['data'])!=template['pkl_sha256']:raise ValueError('Dataset changed')
    with tempfile.TemporaryDirectory() as directory:
        for item in tasks():
            p=recipe(read(stage/'protocols'/item['protocol']/'protocol.json'),item,'preflight-placeholder')
            s=p['models'][p['trainer_alias']]['objectives']['dice']
            if s['max_steps']!=p['budget_flops']//s['flops_per_update'] or s['consumed_training_flops']>p['budget_flops']:
                raise ValueError('Schedule accounting mismatch')
            args=['trainer','--model',p['trainer_alias'],'--objective','dice','--seed',str(item['seed']),
                  '--fold',str(item['fold']),'--pkl',p['data'],'--protocol-dir',str(stage/'protocols'/item['protocol']),
                  '--output',directory]
            with patch.object(sys,'argv',args),patch.object(exp.trainer,'run') as execute:
                exp.trainer.main()
                actual=execute.call_args.args[0]
                if (actual.seed,actual.fold)!=(item['seed'],item['fold']):raise ValueError('CLI mismatch')
    checked=[]
    for model_name in MODELS:
        p=read(stage/'protocols'/f'MSD_{model_name}_f0s7'/'protocol.json')
        exp.base.seed_everything(10)
        model=exp.build_model(p,read(stage/'protocols'/f'MSD_{model_name}_f0s7'/'nnunet_architecture.json'))
        criterion,optimizer=exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],model,p)
        expected='SGD' if model_name=='nnunet' else 'AdamW'
        if type(optimizer).__name__!=expected:raise ValueError('Optimizer changed')
        checked.append(model_name);del optimizer,criterion,model
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    result=subprocess.run(['sbatch','--test-only',str(stage/'run.sbatch'),str(stage)],check=True,capture_output=True,text=True)
    quota=controller.quota()
    if quota['free_gib']<20:raise ValueError('Insufficient quota')
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),
        cli_commands_checked=len(tasks()),models=checked,optimizer_updates=0,quota=quota,
        scheduler_test=result.stdout+result.stderr))


def submit(stage,continuation=None):
    if continuation is not None:raise ValueError('No automatic successor authorized')
    if (stage/'submission.json').exists():raise ValueError('Already submitted; inspect receipt')
    return BASE_SUBMIT(stage)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','verify','preflight','submit','run','worker'])
    parser.add_argument('--stage',type=Path);parser.add_argument('--task',type=json.loads)
    args=parser.parse_args()
    if args.command=='prepare':print(prepare());return
    stage=args.stage.resolve();controller.verify=verify
    if args.command=='worker':worker(stage,args.task)
    elif args.command=='verify':print(verify(stage)['name'])
    elif args.command=='preflight':preflight(stage);print('Seed search preflight passed')
    elif args.command=='submit':print(submit(stage))
    else:run(stage)


if __name__=='__main__':main()
