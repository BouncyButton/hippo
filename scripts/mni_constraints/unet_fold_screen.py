"""Fixed-budget fold-first seed search; never infer quality from seed numbers."""
from copy import deepcopy
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed_recovery as runner
from scripts.mni_constraints.unet_seed10_recovery import ROOT, CAP, NAME as PREVIOUS
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_unet_foldscreen_shared43t_20261108'
MODULE = 'scripts.mni_constraints.unet_fold_screen'
CANDIDATES = list(range(11, 50))
FOLD_ORDER = (1, 0, 2)
RECOVER = ('MNI_basic_unet16_dice_f0s11', 'MNI_basic_unet16_dice_f1s11')


def tasks():
    return [dict(id=f'MNI_basic_unet16_dice_f{f}s{s}',kind='conventional',dataset='MNI',
                 model='basic_unet16',arm='dice',fold=f,seed=s,
                 protocol='basic_unet16_dice',augmentation='none')
            for s in CANDIDATES for f in FOLD_ORDER]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;previous=ROOT/'experiments'/PREVIOUS
    if (stage/'manifest.json').exists():return verify(stage)
    for name,sha in read(previous/'manifest.json').items():assert digest(previous/name)==sha,name
    for part in ('source','data','protocols','known_runs'):
        shutil.copytree(previous/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),stage/'source/scripts/mni_constraints/unet_fold_screen.py')
    for name in ('known_seeds.json','parent_manifest.json','prior_attempts.json'):
        shutil.copy2(previous/name,stage/name)
    shutil.copy2(previous/'manifest.json',stage/'previous_manifest.json')
    old=read(previous/'plan.json')
    rule='Retain previously accepted UNet seeds0,5 and other models0,2,5. New candidates require every fold>=50% at final budget. Reject on the first completed fold below50; skip all unstarted folds. After recovering the two already-started seed11 folds, screen fold1 first, then0,then2, with one active fold per candidate and two candidates in parallel. Stop after the first additional fully qualified seed; finish already-active work for verified archival.'
    plan={**old,'name':NAME,'tasks':tasks(),'num_runs':len(tasks()),'candidate_seeds':CANDIDATES,
          'archive_resume_module':MODULE,'selection_rule':rule,
          'selection_caveat':'Development-validation seed selection; selected-seed metrics are not an unbiased stability estimate. Existing UNet seed5 remains accepted by the previous mean rule; all new qualifiers must pass every fold. Retain compact rejected-candidate evidence.',
          'authorization':'User resumes seed sweep and requests a better search. Earlier user instruction rejects any candidate at a completed result below50. Keep current winning recipes, original data/folds,43T cap; recover seed11 exactly. No larger-data experiment or constraint sweep.',
          'storage':{**old['storage'],'archive':str(ROOT/'reports'/NAME)},
          'fold_order':list(FOLD_ORDER),'recovery_runs':list(RECOVER),
          'protocol_compatibility':'Training protocol bytes retained exactly for checkpoint recovery. Search acceptance and ordering are defined by this plan; its rule supersedes the historical decision_rule text inside the unchanged training protocol.'}
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(previous_manifest_sha256=digest(previous/'manifest.json'),
          changes='New scheduling and qualification controller only; every inherited training source, protocol, split and dataset remains byte-identical.'))
    (stage/'run.sbatch').write_text((previous/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.unet_seed10_recovery',MODULE).replace('hippo-unet-seeds','hippo-unet-screen'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    previous=read(stage/'previous_manifest.json');plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['budget_flops']==CAP and plan['gpu_workers']==2
    for name,sha in previous.items():
        if name.startswith(('source/','data/','protocols/','known_runs/')) and name!='source/source_manifest.json':
            assert digest(stage/name)==sha,name
    assert digest(stage/'known_seeds.json')==previous['known_seeds.json']
    for row in read(stage/'known_seeds.json'):
        for item in row['archives']:
            d=stage/'known_runs'/item['id'];verify_archive(d,read(d/'archive_manifest.json'))
            assert digest(d/'archive_manifest.json')==item['archive_manifest_sha256']
    return plan


def selection(stage,plan):
    rows=deepcopy(read(stage/'known_seeds.json'));decisions=[];winners=[]
    for seed in CANDIDATES:
        values={};ready=[]
        for fold in FOLD_ORDER:
            d=stage/'runs'/f'MNI_basic_unet16_dice_f{fold}s{seed}'
            if not (d/'completion.json').exists():continue
            assert read(d/'completion.json')['status']=='complete'
            values[fold]=100*read(d/'validation_latest.json')['summary']['macro_hard_dice']
            if (d/'cleanup.json').exists() and read(d/'cleanup.json')['status']=='complete':ready.append(fold)
        rejected=any(v<50 for v in values.values())
        qualified=len(ready)==3 and not rejected
        if qualified:winners.append(seed)
        if values:
            decisions.append(dict(seed=seed,completed_folds_pct=values,
                status='rejected' if rejected else 'qualified' if qualified else 'confirming',
                skipped_unstarted_folds=[f for f in FOLD_ORDER if f not in values] if rejected else [],
                mean_validation_pct=sum(values.values())/3 if len(values)==3 else None))
        if len(ready)==3:
            vals=[values[f] for f in range(3)]
            rows.append(dict(model='basic_unet16',seed=seed,validation_by_fold_pct=vals,
                mean_validation_pct=sum(vals)/3,passes_mean50=sum(vals)/3>=50,
                passes_every_fold50=min(vals)>=50))
    selected={m:[0,2,5] for m in ('nnunet','swinunetr','unetr_vit_b16')}
    selected['basic_unet16']=[0,5]+winners[:1]
    return dict(rows=rows,decisions=decisions,selected=selected,complete=bool(winners),
                rule=plan['selection_rule'],caveat=plan['selection_caveat'])


def eligible(stage,plan,state):
    if state['complete']:return []
    rejected={r['seed'] for r in state['decisions'] if r['status']=='rejected'}
    result=[]
    for seed in CANDIDATES:
        if seed in rejected:continue
        group=[t for t in plan['tasks'] if t['seed']==seed]
        missing=[t for t in group if not (stage/'runs'/t['id']/'cleanup.json').exists()]
        if not missing:continue
        # Resume the two saved seed11 trajectories before screening new seeds.
        recovery=[t for t in missing if t['id'] in RECOVER]
        if recovery:
            result.extend(recovery)
        else:
            result.append(missing[0])
    return result


def preflight(stage):
    runner.verify=verify;runner.MODELS=('basic_unet16',)
    return runner.preflight(stage)


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=();runner.verify=verify
    runner.selection=selection;runner.eligible=eligible
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=headroom_bands.train
    batch.run=runner.run
    campaign.main()


if __name__=='__main__':main()
