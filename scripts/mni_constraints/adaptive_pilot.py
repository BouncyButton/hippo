"""Finite four-baseline/eight-band MNI pilot at fixed recipes and FLOP cap."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import time
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, stopping
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands as bands
from scripts.unetr_subfields import cap_extension as batch

ROOT=campaign.ROOT
NAME='mni_adaptive_outer_bands_cap2_f0s0_20261107'
MODULE='scripts.mni_constraints.adaptive_pilot'
PARENT='mni_shared_cap2_baseline_bands_f0s0_20261107'
CAP=63952901334180
MODELS=('nnunet','swinunetr','unetr_vit_b16','basic_unet16')
VARIANTS=('baseline','weighted','adaptive')
RUN_ONE=batch.run_one


def tasks():
    return [dict(id=f'MNI_{m}_{v}_f0s0',kind='conventional',dataset='MNI',model=m,
        arm='dice' if v=='baseline' else 'dice_bands',fold=0,seed=0,protocol=f'{m}_{v}',
        condition=v,augmentation='xy_d4_v1' if m in ('swinunetr','unetr_vit_b16') else 'none',
        target=None if v=='baseline' else bands.TARGETS[m]) for v in VARIANTS for m in MODELS]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME; parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists(): return verify(stage)
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('adaptive_bands.py','test_adaptive_bands.py','profile_adaptive_bands.py',
                     'adaptive_diagnostics.py','adaptive_pilot.py','collect_adaptive.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    source=trainer.read_text(); marker='        training_result = base.evaluate(model, train, device, amp)'
    assert source.count(marker)==1
    source=source.replace(marker,marker+"\n        from scripts.mni_constraints.adaptive_diagnostics import capture\n        capture(model, train, validation, device, amp, args, protocol, ledger['charged_attempts'], ledger['training_flops'], max_steps, native_criterion)")
    trainer.write_text(source)
    references={}; reused=[]
    for task in tasks():
        m,v,arm=task['model'],task['condition'],task['arm']; tag=task['protocol']
        origin=parent/'protocols'/f'{m}_{arm}'; folder=stage/'protocols'/tag
        shutil.copytree(origin,folder); p=read(origin/'protocol.json'); references[tag]=deepcopy(p)
        if v!='baseline':
            profile=read(ROOT/'reports'/NAME/'profiles'/f'{m}_{v}.json')
            assert profile['objective_sha256']==digest(stage/'source/scripts/mni_constraints/objectives.py')
            write(folder/'profile.json',profile)
            s=bands.band_schedule(profile,CAP)
            p['models']={p['trainer_alias']:dict(parameters=profile['parameters'],objectives={arm:s})}
            n=s['max_steps']; p['lr_horizon_updates']=n
            p['constraint_warmup_updates']=s['constraint_warmup_updates']
            p['lr_schedule_details']['horizon_updates']=n
            p['lr_schedule_details']['warmup_updates']=0 if m=='nnunet' else 24 if m=='unetr_vit_b16' else math.ceil(.05*n)
            p['lr_schedule_details']['decay_start_index']=n//2 if p['lr_policy']=='hold_half' else math.ceil(.05*n)
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),profile_sha256=digest(folder/'profile.json'),
            adaptive_variant=v,adaptive_source_sha256=digest(stage/'source/scripts/mni_constraints/adaptive_bands.py'),
            parent_stage=PARENT,parent_protocol_sha256=digest(origin/'protocol.json'),
            spatial_diagnostics='Unaugmented training and validation at25/50/75/100% optimization budget; detached gradient probes on training only.')
        write(folder/'protocol.json',p)
    for m in MODELS:
        for arm in ('dice','dice_bands'):
            archive=ROOT/'reports'/PARENT/'runs'/f'MNI_{m}_{arm}_f0s0'
            verify_archive(archive,read(archive/'archive_manifest.json'))
            assert read(archive/'cleanup.json')['status']=='complete'
            control=stage/'controls'/m/arm; control.mkdir(parents=True)
            for name in ('config.json','training_final.json','validation_latest.json','validation_best.json','metrics.csv','completion.json','runtime_recipe.json','calibration.json'):
                shutil.copy2(archive/name,control/name)
            reused.append(dict(model=m,arm=arm,archive_manifest_sha256=digest(archive/'archive_manifest.json')))
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=12,logical_runs=20,reused_runs=8,
        models=list(MODELS),folds=[0],seeds=[0],budget_flops=CAP,archive_resume_module=MODULE,
        recipe_name='fixed_cap2_recipes_error_weighted_and_locally_expanded_union_bands',
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User accepted error tradeoffs and requested trying this all-four-model adaptive outer-band pilot. Four diagnostic baseline replays, then eight variants. Fold0seed0, fixed cap/recipes, two GPU workers, max one running and one queued allocation.',
        criterion=dict(primary='Final-budget absolute validation macro subfield Dice; differences from replay baseline and archived original bands.',
            secondary='Train/validation curves, per-class and union Dice, error locations and paired transitions. FN/FP tradeoffs accepted.',
            adoption='Single-fold single-seed exploratory pilot; no automatic tuning or expansion.'),
        diagnostic_policy='Baseline predictions retained temporarily outside run folders solely for paired comparisons. Cache manifest archived with baseline spatial reports; delete after both variants are archived. No persistent winning checkpoints.',
        limitation='Adaptive supports on validation are hypothetical coverage diagnostics only. Error coverage by construction does not prove improved generalization.')
    write(stage/'plan.json',plan); write(stage/'control.json',dict(pause=False))
    write(stage/'reference_protocols.json',references); write(stage/'control_reuse.json',reused)
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        trainer_before_sha256=digest(parent/'source/scripts/cross_dataset_campaign/capped_core.py'),
        trainer_after_sha256=digest(trainer),changes='Two detached error-band terms; exact-profile-derived schedules; read-only diagnostic hook preserving RNG/mode/model. Actual base recipes unchanged.'))
    shutil.copy2(ROOT/'reports/mni_outer_band_design_20261107/design.json',stage/'design.json')
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.lower_cap_pairs',MODULE).replace('hippo-mni-budget','hippo-mni-adaptive'))
    campaign.freeze(stage); return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items(): assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json'); references=read(stage/'reference_protocols.json')
    assert plan['tasks']==tasks() and plan['num_runs']==12 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    allowed={'name','data','profile_sha256','models','lr_horizon_updates','constraint_warmup_updates','lr_schedule_details',
             'adaptive_variant','adaptive_source_sha256','parent_stage','parent_protocol_sha256','spatial_diagnostics'}
    for task in tasks():
        folder=stage/'protocols'/task['protocol']; p=read(folder/'protocol.json'); old=references[task['protocol']]
        assert {k for k in set(p)|set(old) if p.get(k)!=old.get(k)}<=allowed
        arm=task['arm']; profile=read(folder/'profile.json'); s=p['models'][p['trainer_alias']]['objectives'][arm]
        expected=batch.schedule(profile['update_flops'],CAP) if arm=='dice' else bands.band_schedule(profile,CAP)
        assert s==expected and p['lr_horizon_updates']==s['max_steps']
        assert campaign.training_flops(s['max_steps'],s)<=CAP<campaign.training_flops(s['max_steps']+1,s)
        assert p['augmentation']==task['augmentation'] and p['folds']==p['seeds']==[0]
        assert p['adaptive_source_sha256']==digest(stage/'source/scripts/mni_constraints/adaptive_bands.py')
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]: assert digest(folder/name)==p[key]
    return plan


def preflight(stage):
    from scripts.mni_constraints import adaptive_bands
    from scripts.mni_constraints.test_adaptive_bands import checks
    scientific=checks(); old=bands.verify; rows=[]
    try:
        bands.verify=verify
        for variant in ('weighted','adaptive'):
            adaptive_bands.install(variant)
            row=bands.preflight(stage,{m:f'{m}_{variant}' for m in MODELS})
            rows.append(dict(variant=variant,**row))
    finally: bands.verify=old
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),scientific=scientific,variants=rows,verified_runs=12,optimizer_updates=0)
    write(stage/'preflight.json',result); return result


def train(args):
    from scripts.mni_constraints import adaptive_bands
    p=read(args.protocol_dir/'protocol.json')
    if p['adaptive_variant']!='baseline': adaptive_bands.install(p['adaptive_variant'])
    bands.train(args)


def run_one(args):
    task=next(t for t in tasks() if t['id']==args.task); stage=args.stage
    required=[]
    if task['condition']!='baseline': required=[t['id'] for t in tasks() if t['condition']=='baseline']
    if task['condition']=='adaptive': required.append(f"MNI_{task['model']}_weighted_f0s0")
    while any(not (stage/'runs'/t/'cleanup.json').exists() for t in required):
        if stopping(): raise SystemExit(75)
        time.sleep(5)
    RUN_ONE(args)
    if task['condition']=='adaptive':
        cache=stage/'diagnostic_cache'/task['model']; deleted=[]
        baseline=stage/'runs'/f"MNI_{task['model']}_baseline_f0s0"
        expected={x['name']:x for q in (25,50,75,100) for x in read(baseline/f'spatial_{q:03d}.json')['baseline_cache']}
        for path in sorted(cache.glob('*.npz')):
            info=expected[path.name]; assert digest(path)==info['sha256'] and path.stat().st_size==info['bytes']
            deleted.append(info); path.unlink()
        assert not list(cache.iterdir()); cache.rmdir()
        write(stage/f"cache_cleanup_{task['model']}.json",dict(status='complete',deleted=deleted,
            reason='Both variant spatial comparisons and metrics archived and acknowledged; temporary baseline prediction dependency fulfilled.'))


def main():
    campaign.MODULE=MODULE; campaign.prepare=prepare; campaign.verify=verify
    campaign.preflight=preflight; campaign.train=train; batch.run_one=run_one
    campaign.main()


if __name__=='__main__': main()
