"""Delayed original UNETR combinations: four-pair pilots then full confirmation."""
from pathlib import Path
import shutil
from statistics import mean
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,headroom_bands,seed_recovery as runner
from scripts.mni_constraints.unetr_delayed_confirm import ROOT,NAME as PARENT,CAP,SEEDS
from scripts.mni_constraints.shared43_study import NAME as ORIGINAL
from scripts.mni_constraints.final53_study import changed_protocol as full_protocol
from scripts.unetr_subfields import cap_extension as batch

NAME='mni_unetr_delayed_combinations53t_20261108'
MODULE='scripts.mni_constraints.unetr_delayed_combinations'
ARMS=('dice_bands_edge','dice_bands_edge_sdf','dice_bands_edge_sdf_depth')
PILOTS=((0,1),(0,2),(5,0),(19,0))  # seed, fold; fixed before running this stage
PAIRS=tuple((s,f) for s in SEEDS for f in range(3))


def tasks():
    pairs=(*PILOTS,*(p for p in PAIRS if p not in PILOTS))
    return [dict(id=f'MNI_unetr_{arm}_f{fold}s{seed}',kind='conventional',dataset='MNI',
        model='unetr_vit_b16',arm=arm,protocol=arm,seed=seed,fold=fold,
        augmentation='xy_d4_v1',pilot=(seed,fold) in PILOTS)
        for seed,fold in pairs for arm in ARMS]


def protocol(old,profile):
    p=full_protocol(old,profile)
    p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),
        primary_metric='Final-budget paired macro Dice gain over the identical53T baseline.',
        decision_rule='Four-pair pilots for three fixed combinations; require all four gains positive, rank by minimum gain then mean then fewer terms. Confirm each qualifying candidate on its five remaining pairs in rank order until one improves on all nine. Retain all attempted cells; no per-fold tuning.',
        selection_caveat='Adaptive development-validation recipe selection, overlapping folds; confirmation means the fixed nine-cell criterion, not independent test generalization.')
    return p


def prepare(_target=None):
    from scripts.mni_constraints.analyze_unetr_delayed_confirm import folder_for
    previous=read(ROOT/'reports'/PARENT/'confirmation_results.json')
    assert previous['status']=='complete', 'Finish current confirmation and its nine baselines first'
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT;original=ROOT/'experiments'/ORIGINAL
    if (stage/'manifest.json').exists():return verify(stage)
    for origin in (parent,original):
        for name,sha in read(origin/'manifest.json').items():assert digest(origin/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('unetr_delayed_combinations.py','final53_study.py','seed_recovery.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    shutil.copy2(original/'manifest.json',stage/'original_manifest.json')
    write(stage/'previous_confirmation.json',previous)
    for arm in ARMS:
        folder=stage/'protocols'/arm
        shutil.copytree(original/'protocols'/('unetr_vit_b16_'+arm),folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        write(folder/'protocol.json',protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json')))
    controls=[]
    for seed,fold in PAIRS:
        origin=folder_for(seed,fold,'baseline')
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        name=f'baseline_f{fold}s{seed}'
        shutil.copytree(origin,stage/'controls'/name)
        controls.append(dict(seed=seed,fold=fold,name=name,archive=str(origin),
            manifest_sha256=digest(origin/'archive_manifest.json'),
            validation=read(origin/'validation_latest.json')['summary']['macro_hard_dice']))
    write(stage/'controls.json',controls)
    plan=dict(name=NAME,tasks=tasks(),num_runs=27,maximum_new_runs=27,pilot_runs=12,
        models=['unetr_vit_b16'],conditions=list(ARMS),seeds=list(SEEDS),folds=[0,1,2],
        pilot_pairs=[list(p) for p in PILOTS],budget_flops=CAP,gpu_workers=2,augmentation='xy_d4_v1',
        conventional_python=read(parent/'plan.json')['conventional_python'],excluded_nodes=['gnode04'],
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User explicitly establishes continuing research until a shared original-constraint UNETR recipe improves every seed0/5/19 and fold0/1/2 pair at the agreed cap. Existing definitions unchanged; finite combination pilots and matched confirmation on two GPUs; other models paused.',
        selection_rule='Require all four pilot gains positive; rank maximum worst-pair gain, then mean gain, then fewer terms. Complete the five remaining pairs for each promising recipe in order; stop this stage on all-nine improvement or exhaustion. Global research continues after exhaustion.',
        evidence='Historical43T seed0 all three folds improved with bands+edge and full combination; delayed bands alone is not consistent at53T. Compare combinations without altering their loss definitions.',
        recipe='All terms target.05 with75% p95 guard; fresh all20 training-crop calibration after102 updates; non-SDF33-update ramp; SDF separately starts nearest5/9 cap; same409-update UNETR base recipe.',
        scope='Development-validation exploration, no seed replacement, no weight or timing tuning within this stage.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Training code byte-identical to completed bands pilot. Select among existing original combinations with fixed delayed calibration/onset and target.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.unetr_delayed_confirm',MODULE).replace('hippo-unetr-confirm','hippo-unetr-combos'))
    original_files=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/PARENT/f.relative_to(stage))
            for f in stage.rglob('*') if f.is_file() and original_files.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/PARENT/'manifest.json')
    for arm in ARMS:
        for f in (stage/'protocols'/arm).iterdir():
            if f.name not in ('protocol.json','parent_protocol.json'):
                donors[str(f.relative_to(stage))]=str(REMOTE/ORIGINAL/'protocols'/('unetr_vit_b16_'+arm)/f.name)
    for c in controls:
        origin=Path(c['archive']);remote_origin=REMOTE/origin.parts[-3]/'runs'/origin.name
        for f in (stage/'controls'/c['name']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':donors[str(f.relative_to(stage))]=str(remote_origin/f.name)
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['num_runs']==27 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    parent=read(stage/'parent_manifest.json');original=read(stage/'original_manifest.json')
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for arm in ARMS:
        folder=stage/'protocols'/arm;p=read(folder/'protocol.json')
        assert digest(folder/'parent_protocol.json')==original[f'protocols/unetr_vit_b16_{arm}/protocol.json']
        assert p==protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'))
        s=p['models']['unet']['objectives'][arm]
        assert s['max_steps']==p['lr_horizon_updates']==409
        assert campaign.training_flops(409,s)<=CAP<campaign.training_flops(410,s)
        assert s['prefix_updates']==102 and s['constraint_warmup_updates']==33
        assert p['calibration_target_ratio']==.05 and p['calibration_p95_cap']==.75
        for name,field in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),
                           ('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[field]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    controls=read(stage/'controls.json');assert {(c['seed'],c['fold']) for c in controls}==set(PAIRS)
    for c in controls:
        folder=stage/'controls'/c['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==c['manifest_sha256']
        cfg=read(folder/'config.json');assert cfg['seed']==c['seed'] and cfg['fold']==c['fold']
        assert cfg['budget_flops']==CAP and cfg['max_steps']==409
        assert cfg['script_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
        assert cfg['initial_lr']==.0006 and cfg['augmentation']=='xy_d4_v1'
        split=read(stage/'protocols'/ARMS[0]/'splits_10patients.json')[c['fold']]
        assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
    return plan


def selection(stage,plan):
    baseline={(c['seed'],c['fold']):c['validation'] for c in read(stage/'controls.json')}
    rows=[]
    for task in plan['tasks']:
        folder=stage/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        val=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        rows.append(dict(**task,validation=val,gain_pp=100*(val-baseline[(task['seed'],task['fold'])])))
    pilots=[r for r in rows if r['pilot']];ranking=[];summaries=[]
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm];pilot=[r for r in group if r['pilot']]
        summary=dict(arm=arm,pairs=len(group),pilot_pairs=len(pilot),
                     minimum_gain_pp=min((r['gain_pp'] for r in group),default=None),
                     mean_gain_pp=mean(r['gain_pp'] for r in group) if group else None,
                     all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group))
        summaries.append(summary)
        if len(pilot)==4 and all(r['gain_pp']>0 for r in pilot):
            ranking.append((min(r['gain_pp'] for r in pilot),mean(r['gain_pp'] for r in pilot),-len(arm.split('_')),arm))
    ordered=[x[-1] for x in sorted(ranking,reverse=True)]
    winner=next((s['arm'] for s in summaries if s['all_nine_improved']),None)
    pending_candidate=next((arm for arm in ordered if sum(r['arm']==arm for r in rows)<9),None)
    finished=len(pilots)==12 and (winner is not None or pending_candidate is None)
    return dict(rows=rows,summary=summaries,pilot_ranking=ordered,
                selected=dict(arm=winner or pending_candidate,winner=winner),complete=finished,
                outcome='nine_pair_success' if winner else 'exhausted' if finished else 'running')


def eligible(stage,plan,state):
    completed={r['id'] for r in state['rows']}
    pilots=[t for t in plan['tasks'] if t['pilot'] and t['id'] not in completed]
    if pilots:return pilots
    if state['complete']:return []
    return [t for t in plan['tasks'] if t['arm']==state['selected']['arm'] and t['id'] not in completed]


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    folder=stage/'protocols'/ARMS[0]
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective=ARMS[0],fold=0))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape)==[1,4,64,64,64]
    assert sum(x.numel() for x in model.parameters())==92685572
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');checks=[]
    for fold,split in enumerate(read(folder/'splits_10patients.json')):
        data=core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
        assert len(data)==20 and len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
        item=data[0];label=item['label'][None]
        p=read(folder/'protocol.json')
        depth=objectives.geometry_weights(label,p['native_geometry'][item['case_name']]['model_affine'],p['depth_geometry'])
        sdf=objectives.signed_distance_maps(label)
        for arm in ARMS:
            torch.manual_seed(731);z=torch.randn((1,4,64,64,64),requires_grad=True)
            row=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
            weights,stats=objectives.select_weights([row],arm,.05,.75)
            assert all(s['weighted_median_ratio']<=.0500001 and s['weighted_p95_ratio']<=.7500001 for s in stats.values())
            value=objectives.training_loss(z,label,arm,weights,1.,sdf,None,depth,1.,identity='unetr_vit_b16')
            assert torch.isfinite(value) and torch.isfinite(torch.autograd.grad(value,z)[0]).all()
            checks.append(dict(fold=fold,arm=arm,synthetic_statistics=stats))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,
                checks=checks,unchanged_training_runtime=True,baselines=9)
    write(stage/'preflight.json',result);return result


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=()
    runner.verify,runner.selection,runner.eligible=verify,selection,eligible
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train
    batch.run=runner.run
    campaign.main()


if __name__=='__main__':main()
