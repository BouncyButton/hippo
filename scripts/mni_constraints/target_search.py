"""Finite fold0/seed0 target search for bands and bands+edge on four models."""
import json
from copy import deepcopy
from pathlib import Path
import shutil
from statistics import mean

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.mni_constraints import strong_pilot as pilot, campaign as original

ROOT=original.ROOT
NAME='mni_constraint_target_search_f0s0_20261107'
MODULE='scripts.mni_constraints.target_search'
RECOVERY=pilot.NAME
TARGETS={m:([.05,.1,.2,.4,.6,.8] if m=='nnunet' else [.05,.1,.2,.4,.6]) for m in original.batch.MODELS}
PILOT_PREFLIGHT=pilot.preflight
SUMMARIZE=controller.summarize


def tag(target):return f't{round(100*target):02d}'


def all_tasks():
    rows=[]
    # Prioritize60% for nnUNet and lower targets for the other models.
    priority={'nnunet':[.6,.4,.8,.2,.1,.05],**{m:[.2,.05,.1,.4,.6] for m in original.batch.MODELS if m!='nnunet'}}
    for index in range(6):
        for arm in pilot.ARMS:
            for model in original.batch.MODELS:
                if index>=len(priority[model]):continue
                target=priority[model][index]
                rows.append(dict(id=f'MNI_{model}_{arm}_{tag(target)}_f0s0',kind='conventional',dataset='MNI',model=model,
                    arm=arm,fold=0,seed=0,target=target,protocol=f'{model}_{arm}_{tag(target)}',
                    augmentation='xy_d4_v1' if model=='unetr_vit_b16' else 'none'))
    return rows


def prepare():
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/RECOVERY
    if (stage/'manifest.json').exists():return verify(stage)
    stage.mkdir(parents=True,exist_ok=True)
    for part in ('source','data','retirement_protocols','protocols'):
        shutil.copytree(parent/part,stage/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('target_search.py','launch_search.py','collect_search.py','launch_strong.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    shutil.copy2(ROOT/'scripts/unetr_subfields/mni_diagnostic.py',stage/'source/scripts/unetr_subfields/mni_diagnostic.py')
    plan=deepcopy(read(parent/'plan.json'));reuse=[];fixtures={}
    for model in original.batch.MODELS:
        root=ROOT/'reports'/original.NAME/'runs'/f'MNI_{model}_dice_bands_f0s0'
        calibration=read(root/'calibration.json');statistics=calibration['statistics']['bands']
        recomputed=min(.1/statistics['median_raw_ratio'],.75/statistics['p95_raw_ratio'])
        if abs(recomputed-calibration['weights']['bands'])<1e-14:
            from scripts.cross_dataset_campaign.storage import verify_archive
            verify_archive(root,read(root/'archive_manifest.json'))
            reuse.append(dict(model=model,arm='dice_bands',target=.1,fold=0,seed=0,
                id=f'MNI_{model}_dice_bands_t10_f0s0',archive=str(root),archive_manifest_sha256=digest(root/'archive_manifest.json'),
                final_dice=read(root/'validation_latest.json')['summary']['macro_hard_dice'],
                explanation='Original10%/50% calibration yields exactly the same coefficient under the new75% guard; other numerical protocol fields unchanged.'))
        fixtures[model]=read(ROOT/'reports'/original.NAME/'runs'/f'MNI_{model}_dice_bands_edge_sdf_depth_f0s0'/'calibration.json')
    used={r['id'] for r in reuse};work=[r for r in all_tasks() if r['id'] not in used]
    for task in all_tasks():
        folder=stage/'protocols'/task['protocol']
        source=parent/'protocols'/(task['model']+'_'+task['arm'])
        shutil.copytree(source,folder,dirs_exist_ok=True)
        p=read(folder/'protocol.json')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),calibration_target_ratio=task['target'],
            calibration_p95_cap=1. if task['target']==.8 else .75,
            calibration_strength_policy='Per-model target grid;75% p95 guard through60% targets,100% guard for80% target. Recalibrate each term sequentially using training-only gradients.',
            selection_policy='Select best tested per-model target by mean final-budget validation Dice across bands and bands+edge at fixedfold0seed0; also report each arm separately.')
        write(folder/'protocol.json',p)
    # The eight untagged protocols are only preflight fixtures, never run tasks.
    for folder in (stage/'protocols').iterdir():
        if folder.name.endswith(('_dice_bands','_dice_bands_edge')):
            p=read(folder/'protocol.json');p['data']=str(REMOTE/NAME/'data/MNI.pkl');write(folder/'protocol.json',p)
    plan.update(name=NAME,tasks=work,num_runs=len(work),logical_cells=46,search_cells=42,
        calibration_target_ratio='model_specific_grid',target_grid=TARGETS,archive_resume_module=MODULE,
        calibration_p95_policy='75% for targets5..60%;100% for nnUNet80%',
        reused_constraint_runs=len(reuse),
        storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)),
        selection_policy='Best tested common target per model maximizes mean final-budget Dice across both arms on fold0seed0; ties choose the smaller target. Per-arm winners also reported. No expansion of folds or seeds.',
        authorization='User requested finding best gradient target per model, noting nnUNet60% and lower targets for others; retain requested bands/bands+edge,onefold/oneseed,twoGPUworkers and cleanup.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'constraint_reuse.json',reuse);write(stage/'calibration_fixtures.json',fixtures)
    shutil.copy2(parent/'baseline_reuse.json',stage/'baseline_reuse.json')
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(ROOT/'experiments'/original.NAME/'manifest.json'),
        prepared_pilot_manifest_sha256=digest(parent/'manifest.json'),changes='Broaden not-yet-submitted60% pilot into a finite per-model target grid. Same objective, budgets, base losses, data and schedules. Training starts after abandoned-run archive and cleanup.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('hippo-mni-target60','hippo-mni-target-search').replace('scripts.mni_constraints.strong_pilot',MODULE))
    original.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    assert f'-m {MODULE} run --stage' in (stage/'run.sbatch').read_text(), 'Wrong batch entry point'
    assert (stage/'source/scripts/unetr_subfields/mni_diagnostic.py').is_file(), 'Missing retirement archive helper'
    plan=read(stage/'plan.json');reuse=read(stage/'constraint_reuse.json');used={r['id'] for r in reuse}
    assert plan['tasks']==[r for r in all_tasks() if r['id'] not in used]
    assert plan['num_runs']+len(reuse)==42 and plan['gpu_workers']==2
    assert plan['target_grid']==TARGETS and plan['folds']==plan['seeds']==[0]
    for task in all_tasks():
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        assert p['calibration_target_ratio']==task['target'] and p['calibration_p95_cap']==(1. if task['target']==.8 else .75)
        assert p['folds']==p['seeds']==[0] and p['budget_flops']==original.CAP
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[key]
        s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
        assert p['lr_horizon_updates']==s['max_steps']
        assert original.training_flops(s['max_steps'],s)<=original.CAP<original.training_flops(s['max_steps']+1,s)
    return plan


def preflight(stage):
    from scripts.unetr_subfields.mni_diagnostic import archive_and_retire
    assert callable(archive_and_retire)
    pilot.verify=verify
    result=PILOT_PREFLIGHT(stage)
    result['status']='checking_target_grid';write(stage/'preflight.json',result)
    from scripts.mni_constraints import objectives
    checks=[]
    fixtures=read(stage/'calibration_fixtures.json')
    for task in all_tasks():
        cap=1. if task['target']==.8 else .75
        weights,stats=objectives.select_weights(deepcopy(fixtures[task['model']]['cases']),task['arm'],task['target'],cap)
        assert all(v>0 for v in weights.values())
        assert all(v['weighted_median_ratio']<=task['target']+1e-8 and v['weighted_p95_ratio']<=cap+1e-8 for v in stats.values())
        checks.append(dict(model=task['model'],arm=task['arm'],target=task['target'],weights=weights,statistics=stats))
    result.update(status='passed',target_grid_checks=checks,source_calibration_cases_used_only_for_numerical_preflight=True)
    write(stage/'preflight.json',result);return result


def comparison(stage,plan):
    rows=[dict(model=r['model'],arm=r['arm'],target=r['target'],final_dice=r['final_dice'],reused=True) for r in read(stage/'constraint_reuse.json')]
    baseline={r['model']:r['final_dice'] for r in read(stage/'baseline_reuse.json')}
    for task in plan['tasks']:
        folder=stage/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        rows.append(dict(model=task['model'],arm=task['arm'],target=task['target'],
            final_dice=read(folder/'validation_latest.json')['summary']['macro_hard_dice'],reused=False))
    for row in rows:row['delta_pp']=100*(row['final_dice']-baseline[row['model']])
    recommendations={}
    for model,targets in TARGETS.items():
        group=[r for r in rows if r['model']==model]
        if len(group)!=2*len(targets):continue
        ranking=sorted([dict(target=t,mean_dice=mean(r['final_dice'] for r in group if r['target']==t)) for t in targets],key=lambda r:(-r['mean_dice'],r['target']))
        recommendations[model]=dict(best_tested_target=ranking[0]['target'],ranking=ranking,
            per_arm={arm:max((r for r in group if r['arm']==arm),key=lambda r:(r['final_dice'],-r['target'])) for arm in pilot.ARMS})
    return dict(status='complete' if len(rows)==42 else 'partial',cells_complete=len(rows),cells_planned=42,
        metric='Final-budget validation macro Dice; pilotfold0seed0 only, not a confirmed3fold3seed optimum.',
        recommendations=recommendations,results=rows)


def summarize(stage,plan,status,active=None):
    SUMMARIZE(stage,plan,status,active)
    state=read(stage/'status.json');state['target_search']=comparison(stage,plan);write(stage/'status.json',state)
    if status=='complete':write(stage/'target_comparison.json',state['target_search'])


def main():
    pilot.NAME=NAME;pilot.MODULE=MODULE;pilot.prepare=prepare;pilot.verify=verify;pilot.preflight=preflight
    controller.summarize=summarize
    pilot.main()


if __name__=='__main__':main()
