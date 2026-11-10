"""Fresh baseline/bands pairs at a2x shared cap, with fixed model recipes."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign import controller
from scripts.mni_constraints import campaign,headroom_bands as bands
from scripts.unetr_subfields import cap_extension as batch

ROOT=campaign.ROOT
NAME='mni_shared_cap2_baseline_bands_f0s0_20261107'
MODULE='scripts.mni_constraints.lower_cap_pairs'
BASE_PARENT='mni_shared_cap3_headroom_3fold3seed_20261107'
BANDS_PARENT=bands.NAME
CAP=63952901334180
ARMS=('dice','dice_bands')
SUMMARIZE=controller.summarize


def tasks(protocols):
    return [dict(id=f'MNI_{model}_{arm}_f0s0',kind='conventional',dataset='MNI',
                 model=model,arm=arm,fold=0,seed=0,protocol=f'{model}_{arm}',
                 augmentation=protocols[f'{model}_{arm}']['augmentation'],
                 target=bands.TARGETS[model] if arm=='dice_bands' else None)
            for model in batch.MODELS for arm in ARMS]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/BANDS_PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('lower_cap_pairs.py','headroom_bands.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    protocols={};references={}
    for model in batch.MODELS:
        for arm in ARMS:
            old_stage=BASE_PARENT if arm=='dice' else BANDS_PARENT
            origin=ROOT/'experiments'/old_stage/'protocols'/model
            tag=f'{model}_{arm}';folder=stage/'protocols'/tag
            shutil.copytree(origin,folder);p=deepcopy(read(origin/'protocol.json'));references[tag]=deepcopy(p)
            profile=read(folder/'profile.json')
            s=batch.schedule(profile['update_flops'],CAP) if arm=='dice' else bands.band_schedule(profile,CAP)
            alias=p['trainer_alias'];n=s['max_steps'];warmup=0 if model=='nnunet' else 24 if model=='unetr_vit_b16' else math.ceil(.05*n)
            p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),budget_flops=CAP,folds=[0],seeds=[0],
                     models={alias:dict(parameters=profile['parameters'],objectives={arm:s})},
                     requested_arm_order=[arm],lr_horizon_updates=n,
                     constraint_warmup_updates=s['constraint_warmup_updates'],
                     validation_target_flops=[CAP*i//20 for i in range(1,21)],
                     budget_origin='Prospective2x historical common optimization cap, one third below the preceding3x paired experiment.',
                     checkpoint_selection='Final-budget checkpoint primary. Fresh runs with cap-matched schedules; no per-model or validation-selected stopping.',
                     learning_rate_policy='Unchanged peak, optimizer and decay family; schedule horizon matched to the new2x cap.',
                     lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=n,warmup_updates=warmup,
                                              decay_start_index=n//2 if p['lr_policy']=='hold_half' else warmup),
                     parent_protocol_sha256=digest(origin/'protocol.json'),parent_stage=old_stage)
            if arm=='dice_bands':
                p['timing_policy']='Same10% baseline-only prefix and10% ramp rule, recomputed for the2x cap. Targets and75% p95 guard unchanged.'
            write(folder/'protocol.json',p);protocols[tag]=p
    plan=deepcopy(read(parent/'plan.json'))
    plan.pop('prerequisite_stage',None)
    plan.update(name=NAME,tasks=tasks(protocols),num_runs=8,logical_runs=8,reused_runs=0,
                budget_flops=CAP,multiplier=2,archive_resume_module=MODULE,
                recipe_name='fixed_recipes_fresh_baseline_bands_at_2x',
                storage=dict(minimum_free_gib=12,estimated_peak_mirrored_gib=10,archive=str(ROOT/'reports'/NAME)),
                authorization='User requested testing a lower budget after examining early bands gains and late overfitting. Eight fresh paired runs: four models,baseline and bands,fold0seed0. Keep model recipes,augmentation,gradient targets and splits fixed. Two GPU workers and at most one running plus one queued job.',
                criterion=dict(primary='Final-budget paired validation Dice gain; report absolute validation Dice and whether each run exceeds50%.',
                               secondary='Training Dice, train/validation gaps and curves. A larger bands-minus-baseline difference alone does not establish a better absolute model.',
                               adoption='Exploratory cap test; do not replace the established3x recipes or expand folds/seeds automatically.'),
                limitation='Fresh shorter schedules differ from truncating3x curves. A lower cap can undertrain slower models; report failures without changing recipes during the comparison.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'reference_protocols.json',references)
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Only common budget and its derived LR/constraint timing change. All eight runs restart from fixedseed0; no3x baselines are reused as2x controls. Existing losses,optimizer settings,augmentation,data and band targets preserved.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(bands.MODULE,MODULE).replace('hippo-mni-bands','hippo-mni-budget'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');references=read(stage/'reference_protocols.json')
    protocols={tag:read(stage/'protocols'/tag/'protocol.json') for tag in references}
    assert plan['tasks']==tasks(protocols) and plan['num_runs']==8 and plan['reused_runs']==0 and plan['gpu_workers']==2
    assert plan['budget_flops']==CAP
    permitted={'name','data','budget_flops','folds','seeds','models','requested_arm_order','lr_horizon_updates',
               'constraint_warmup_updates','validation_target_flops','budget_origin','checkpoint_selection',
               'learning_rate_policy','lr_schedule_details','parent_protocol_sha256','parent_stage','timing_policy'}
    for task in plan['tasks']:
        tag=task['protocol'];p=protocols[tag];old=references[tag];folder=stage/'protocols'/tag;profile=read(folder/'profile.json')
        assert {k for k in set(p)|set(old) if p.get(k)!=old.get(k)}<=permitted
        arm=task['arm'];s=p['models'][p['trainer_alias']]['objectives'][arm]
        expected=batch.schedule(profile['update_flops'],CAP) if arm=='dice' else bands.band_schedule(profile,CAP)
        assert s==expected and p['budget_flops']==CAP and p['lr_horizon_updates']==s['max_steps']
        assert campaign.training_flops(s['max_steps'],s)<=CAP<campaign.training_flops(s['max_steps']+1,s)
        assert p['folds']==p['seeds']==[0] and p['constraints']==([] if arm=='dice' else ['bands'])
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[key]
        n=s['max_steps'];lr=p['initial_lrs'][p['trainer_alias']]
        assert batch.lr_at(n+10,n,lr,task['model'],p['lr_policy'])<=lr*.010001
    return plan


def preflight(stage):
    plan=verify(stage);original=bands.verify
    try:
        bands.verify=verify
        result=bands.preflight(stage,{m:f'{m}_dice_bands' for m in batch.MODELS})
    finally:bands.verify=original
    result.update(verified_fresh_runs=len(plan['tasks']),paired_recipe_differences='Only cap and derived schedule fields; exact protocol comparison passed.')
    write(stage/'preflight.json',result);return result


def summarize(stage,plan,status,active=None):
    SUMMARIZE(stage,plan,status,active);pairs=[]
    for model in batch.MODELS:
        rows={}
        for arm in ARMS:
            folder=stage/'runs'/f'MNI_{model}_{arm}_f0s0'
            if not (folder/'cleanup.json').exists():continue
            rows[arm]=dict(training_dice=read(folder/'training_final.json')['summary']['macro_hard_dice'],
                           validation_dice=read(folder/'validation_latest.json')['summary']['macro_hard_dice'])
        if len(rows)==2:
            pairs.append(dict(model=model,**rows,delta_pp=100*(rows['dice_bands']['validation_dice']-rows['dice']['validation_dice'])))
    state=read(stage/'status.json');state['paired_comparison']=pairs;write(stage/'status.json',state)
    if status=='complete':write(stage/'paired_comparison.json',pairs)


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=bands.train;controller.summarize=summarize
    campaign.main()


if __name__=='__main__':main()
