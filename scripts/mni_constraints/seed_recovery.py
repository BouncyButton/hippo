"""Finite, seed-major baseline recovery without changing the accepted recipes."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, install_signals, stopping
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import quota, retire, verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.shared43_study import NAME as PARENT, CAP
from scripts.unetr_subfields import cap_extension as batch
from scripts.unetr_subfields.contracts import ROOT

NAME = 'mni_seed_recovery_shared43t_20261108'
MODULE = 'scripts.mni_constraints.seed_recovery'
MODELS = ('nnunet', 'basic_unet16')
CANDIDATES = list(range(3, 12))
RETIREMENTS = ('MNI_swinunetr_dice_bands_depth_f1s2', 'MNI_nnunet_dice_bands_depth_f1s2')
ORIGINAL_RUN_ONE = batch.run_one


def tasks():
    return [dict(id=f'MNI_{m}_dice_f{f}s{s}', kind='conventional', dataset='MNI',
                 model=m, arm='dice', fold=f, seed=s, protocol=m+'_dice', augmentation='none')
            for s in CANDIDATES for m in MODELS for f in range(3)]


def prepare(_target=None):
    stage = ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():
        return verify(stage)
    parent = ROOT/'experiments'/PARENT
    for name, sha in read(parent/'manifest.json').items():
        assert digest(parent/name) == sha, name
    for part in ('source', 'data'):
        shutil.copytree(parent/part, stage/part, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__), stage/'source/scripts/mni_constraints/seed_recovery.py')
    shutil.copy2(ROOT/'scripts/mni_constraints/collect_seed_recovery.py', stage/'source/scripts/mni_constraints/collect_seed_recovery.py')
    for m in MODELS:
        folder = stage/'protocols'/(m+'_dice')
        shutil.copytree(parent/'protocols'/(m+'_dice'), folder)
        shutil.copy2(folder/'protocol.json', folder/'parent_protocol.json')
        p = read(folder/'protocol.json')
        p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), seeds=CANDIDATES,
                 decision_rule='Evaluate all three original folds for every numerical candidate; mean final-budget Dice >=50%; report each fold and prefer every-fold >=50%. Preserve failed attempts.',
                 primary_metric='Final-budget foreground macro Dice averaged equally across folds0,1,2 for seed qualification.')
        write(folder/'protocol.json', p)
    parent_tasks = read(parent/'plan.json')['tasks']
    for identifier in RETIREMENTS:
        t = next(t for t in parent_tasks if t['id'] == identifier)
        shutil.copytree(parent/'protocols'/t['protocol'], stage/'retirement_protocols'/identifier)
    shutil.copy2(parent/'manifest.json', stage/'parent_manifest.json')
    handoff = read(ROOT/'reports/mni_seed_recovery_handoff_20261108.json')
    # Freeze the actual source archives behind all twelve original seed decisions.
    known = []
    for row in handoff['seed_evidence']:
        archives = []
        for fold in range(3):
            identifier = f"MNI_{row['model']}_dice_f{fold}s{row['seed']}"
            source = ROOT/'reports'/PARENT/('frozen/reused_runs' if fold == row['seed'] == 0 else 'runs')/identifier
            verify_archive(source, read(source/'archive_manifest.json'))
            assert read(source/'cleanup.json')['status'] == 'complete'
            actual = read(source/'validation_latest.json')['summary']['macro_hard_dice']*100
            assert abs(actual-row['validation_by_fold_pct'][fold]) < 1e-9
            dest = stage/'known_runs'/identifier
            shutil.copytree(source, dest)
            archives.append(dict(id=identifier, archive_manifest_sha256=digest(dest/'archive_manifest.json')))
        known.append(dict(**row, archives=archives))
    write(stage/'known_seeds.json', known)
    plan = dict(name=NAME, tasks=tasks(), num_runs=len(tasks()), retirements=list(RETIREMENTS),
                parent_stage=str(REMOTE/PARENT), models=list(MODELS), candidate_seeds=CANDIDATES,
                budget_flops=CAP, gpu_workers=2, augmentation='none', folds=[0,1,2],
                conventional_python=handoff['remote']['python'], graceful_stop_seconds=20700,
                archive_wait_timeout_seconds=600, archive_resume_module=MODULE,
                storage=dict(minimum_free_gib=16, estimated_peak_mirrored_gib=12, archive=str(ROOT/'reports'/NAME)),
                selection_rule='Historical mean>=50%; rank qualifying seeds by every-fold>=50% then numerical seed. Stop each model after three qualifiers. All attempted folds retained.',
                authorization='Recover three working baseline seeds per model; start numerical candidates3; retain fixed folds, recipes,10 subjects and43T cap. Retire two abandoned runs with zero updates. Two GPU workers, maxone running and one queued allocation; per-run verified archival and cleanup.',
                selection_caveat='Development-validation selection, not an untouched test estimate. Never replace folds or remove failed attempts.')
    write(stage/'plan.json', plan)
    write(stage/'control.json', dict(pause=False))
    write(stage/'source_version.json', dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Training runtime, data, loss, LR, augmentation, cap and splits unchanged. New finite adaptive seed scheduler and zero-update retirement only.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.shared43_study', MODULE).replace('hippo-mni-14', 'hippo-mni-seeds'))
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name, sha in read(stage/'manifest.json').items():
        assert digest(stage/name) == sha, name
    plan = read(stage/'plan.json')
    assert plan['tasks'] == tasks() and plan['budget_flops'] == CAP and plan['gpu_workers'] == 2
    parent_manifest = read(stage/'parent_manifest.json')
    allowed = {'name', 'data', 'seeds', 'decision_rule', 'primary_metric'}
    for m in MODELS:
        folder = stage/'protocols'/(m+'_dice')
        p = read(folder/'protocol.json')
        old = read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json') == parent_manifest[f'protocols/{m}_dice/protocol.json']
        assert {k:v for k,v in p.items() if k not in allowed} == {k:v for k,v in old.items() if k not in allowed}
        # Original protocol included among immutable known archive configuration evidence.
        original = read(stage/'known_runs'/f'MNI_{m}_dice_f1s0'/'config.json')
        assert p['models'][p['trainer_alias']]['objectives']['dice'] == original['schedule']
        assert p['initial_lrs'][p['trainer_alias']] == original['initial_lr']
        assert p['lr_horizon_updates'] == original['lr_horizon_updates']
        assert p['augmentation'] == original['augmentation'] == 'none'
        assert p['budget_flops'] == original['budget_flops'] == CAP
        assert p['folds'] == [0,1,2] and p['seeds'] == CANDIDATES
        for name, key in [('profile.json','profile_sha256'), ('splits_10patients.json','splits_sha256'), ('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name) == p[key] == parent_manifest[f'protocols/{m}_dice/{name}']
        assert digest(stage/'data/MNI.pkl') == p['pkl_sha256']
    # Every inherited training source remains exactly the parent's frozen source.
    for name, sha in parent_manifest.items():
        if name.startswith('source/') and name not in ('source/source_manifest.json',):
            assert digest(stage/name) == sha, name
    for row in read(stage/'known_seeds.json'):
        for item in row['archives']:
            folder = stage/'known_runs'/item['id']
            assert digest(folder/'archive_manifest.json') == item['archive_manifest_sha256']
            verify_archive(folder, read(folder/'archive_manifest.json'))
    return plan


def preflight(stage):
    verify(stage)
    import torch, monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4)
    assert monai.__version__ == '1.6.0'
    frame = core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    rows = []
    for m in MODELS:
        folder = stage/'protocols'/(m+'_dice'); p = read(folder/'protocol.json')
        batch.bind(SimpleNamespace(protocol_dir=folder, model=p['trainer_alias'], objective='dice', fold=0))
        model = core.base.build_model(p['trainer_alias'], read(folder/'nnunet_architecture.json')).to('meta').train()
        outputs = model(torch.zeros((1,1,64,64,64), device='meta'))
        shapes = [list(x.shape) for x in outputs] if isinstance(outputs,(list,tuple)) else [list(outputs.shape)]
        profile = read(folder/'profile.json')
        assert shapes == profile['output_shapes']
        assert sum(x.numel() for x in model.parameters()) == profile['parameters']
        for split in read(folder/'splits_10patients.json'):
            assert len(core.PreparedSubfields(frame, split['train'], DATASETS['MNI'])) == 20
            assert len(core.PreparedSubfields(frame, split['val'], DATASETS['MNI'])) == 10
        rows.append(dict(model=m, output_shapes=shapes, updates=p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']))
    result = dict(status='passed', manifest_sha256=digest(stage/'manifest.json'), checks=rows,
                  optimizer_updates=0, torch=torch.__version__, monai=monai.__version__)
    write(stage/'preflight.json',result)
    return result


def selection(stage, plan):
    rows = deepcopy(read(stage/'known_seeds.json'))
    for m in MODELS:
        for seed in CANDIDATES:
            group = [t for t in plan['tasks'] if t['model']==m and t['seed']==seed]
            if not all((stage/'runs'/t['id']/'cleanup.json').exists() for t in group):
                continue
            values = [100*read(stage/'runs'/t['id']/'validation_latest.json')['summary']['macro_hard_dice'] for t in group]
            rows.append(dict(model=m,seed=seed,validation_by_fold_pct=values,
                             mean_validation_pct=sum(values)/3,passes_mean50=sum(values)/3>=50,
                             passes_every_fold50=min(values)>=50))
    selected = {}
    for m in {r['model'] for r in rows}:
        passing = sorted((r for r in rows if r['model']==m and r['passes_mean50']),
                         key=lambda r:(not r['passes_every_fold50'],r['seed']))
        selected[m] = [r['seed'] for r in passing[:3]]
    return dict(rows=rows, selected=selected, complete=all(len(s)==3 for s in selected.values()),
                rule=plan['selection_rule'], caveat=plan['selection_caveat'])


def eligible(stage, plan, state):
    """Only one candidate per model at a time; never omit a failed candidate's folds."""
    result = []
    for m in MODELS:
        if len(state['selected'][m]) >= 3:
            continue
        for seed in CANDIDATES:
            group = [t for t in plan['tasks'] if t['model']==m and t['seed']==seed]
            missing = [t for t in group if not (stage/'runs'/t['id']/'cleanup.json').exists()]
            if missing:
                result.extend(missing)
                break
    return result


def retire_interrupted(stage, identifier):
    """Evaluate relocated recovery files; do not run an optimizer or alter checkpoints."""
    import torch
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.unetr_subfields.mni_diagnostic import manifest
    torch.set_num_threads(4)
    output=stage/'runs'/identifier
    if (output/'cleanup.json').exists(): return
    if not (output/'archive_manifest.json').exists():
        folder=stage/'retirement_protocols'/identifier
        p=read(folder/'protocol.json');cfg=read(output/'config.json')
        assert cfg['protocol_sha256']==digest(folder/'protocol.json')
        cp=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        assert cp['configuration']==cfg
        if (output/'budget_ledger.json').exists():
            ledger=read(output/'budget_ledger.json')
            assert cp['charged_attempts']<=ledger['charged_attempts']
        else:
            assert cp['step']==cp['charged_attempts']==0, 'Only a zero-step checkpoint may lack a ledger'
            ledger=dict(charged_attempts=0,training_flops=0,calibration_flops=0)
        before={f.name:digest(f) for f in output.glob('*.pt')}
        batch.bind(SimpleNamespace(protocol_dir=folder,model=p['trainer_alias'],objective=cfg['objective'],fold=cfg['fold']))
        device=torch.device('cuda');assert torch.cuda.is_available()
        model=core.base.build_model(p['trainer_alias'],read(folder/'nnunet_architecture.json')).to(device)
        model.load_state_dict(cp['model'])
        assert digest(stage/'data/MNI.pkl')==cfg['dataset_sha256']
        frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');spec=DATASETS['MNI']
        for split,key in [('training','train_cases'),('validation','validation_cases')]:
            report=core.evaluate(model,core.PreparedSubfields(frame,cfg[key],spec),device,True,spec['classes'])
            assert {r['case_name'] for r in report['cases']}==set(cfg[key])
            assert all('confusion' in r and 'foreground_fn' in r for r in report['cases'])
            write(output/(split+'_interrupted.json'),report)
        if (output/'checkpoint_best.pt').exists():
            best=torch.load(output/'checkpoint_best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(best['model'])
            write(output/'validation_best_poststop.json',core.evaluate(model,core.PreparedSubfields(frame,cfg['validation_cases'],spec),device,True,spec['classes']))
        assert all(digest(output/n)==sha for n,sha in before.items())
        write(output/'interrupted_protocol.json',p)
        write(output/'abandonment.json',dict(status='abandoned_by_user',original_run=identifier,
              original_stage=str(REMOTE/PARENT),completed_updates=cp['step'],charged_updates=ledger['charged_attempts'],
              consumed_training_flops=ledger['training_flops'],additional_optimizer_updates=0,
              missing_ledger_zero_step=not (output/'budget_ledger.json').exists(),
              excluded_from_complete_results=True,ap_swaps_applicable=False,
              reason='Full14-condition study stopped for baseline seed recovery and UNETR strength investigation.',
              parent_manifest_sha256=digest(stage/'parent_manifest.json')))
        manifest(output,dict(id=identifier,kind='abandoned',dataset='MNI'))
    start=time.monotonic()
    while not (output/'archive_acknowledgment.json').exists():
        if stopping() or time.monotonic()-start>600:raise SystemExit(75)
        time.sleep(5)
    retire(output)


def run_one(args):
    if args.task in RETIREMENTS:
        retire_interrupted(args.stage,args.task)
    else:
        ORIGINAL_RUN_ONE(args)


def run(stage):
    plan=verify(stage);job=os.environ['SLURM_JOB_ID'];install_signals()
    os.environ['HIPPO_CONTROL']=str(stage/'control.json')
    os.environ['HIPPO_DEADLINE']=str(time.time()+plan['graceful_stop_seconds'])
    with (stage/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if stopping():return
        controller.submit(stage,continuation=job)
        devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2
        active={};failed=None
        try:
            while True:
                for device,(identifier,child) in list(active.items()):
                    code=child.poll()
                    if code is not None:
                        del active[device]
                        if code not in (0,75):failed=dict(task=identifier,exit_code=code)
                        if code==75:os.environ['HIPPO_DEADLINE']='0'
                state=selection(stage,plan);write(stage/'selection.json',state)
                retiring=[i for i in RETIREMENTS if not (stage/'runs'/i/'cleanup.json').exists()]
                pending=retiring if retiring else [t['id'] for t in eligible(stage,plan,state)]
                pending=[i for i in pending if i not in {i for i,_ in active.values()}]
                if stopping() or failed:
                    for _,child in active.values():
                        if child.poll() is None:child.send_signal(signal.SIGUSR1)
                    if not active:break
                else:
                    for device in devices:
                        if device not in active and pending:
                            assert quota()['free_gib']>=plan['storage']['minimum_free_gib']
                            identifier=pending.pop(0)
                            child=subprocess.Popen([plan['conventional_python'],'-u','-m',MODULE,'run-one','--stage',str(stage),'--task',identifier],
                                env={**os.environ,'CUDA_VISIBLE_DEVICES':device},cwd=stage/'source')
                            active[device]=(identifier,child)
                    if not pending and not active:break
                write(stage/'status.json',dict(status='running',active=[i for i,_ in active.values()],
                      completed=sum((stage/'runs'/t['id']/'cleanup.json').exists() for t in plan['tasks']),
                      total=plan['num_runs'],selected=state['selected'],job_id=job))
                time.sleep(5)
            if failed:raise RuntimeError(str(failed))
            status=('paused' if read(stage/'control.json')['pause'] else 'continuing_at_time_limit') if stopping() else ('complete' if state['complete'] else 'candidate_block_exhausted')
            write(stage/'status.json',dict(status=status,active=[],selected=state['selected'],job_id=job,
                  completed=sum((stage/'runs'/t['id']/'cleanup.json').exists() for t in plan['tasks']),total=plan['num_runs']))
            if status!='continuing_at_time_limit':controller.cancel_successor(stage,job)
        except Exception:
            os.environ['HIPPO_DEADLINE']='0'
            for _,child in active.values():
                if child.poll() is None:child.send_signal(signal.SIGUSR1)
            for _,child in active.values():child.wait()
            controller.cancel_successor(stage,job)
            write(stage/'failure.json',dict(traceback=traceback.format_exc()))
            write(stage/'status.json',dict(status='failed',job_id=job))
            raise


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=headroom_bands.train
    batch.run=run;batch.run_one=run_one
    campaign.main()


if __name__=='__main__':main()
