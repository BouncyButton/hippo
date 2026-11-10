"""Conditional full MNI matrix, gated by all nine delayed-UNETR paired gains."""
from copy import deepcopy
import math
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE, ARMS
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.common53_study import ROOT, NAME as PARENT, CAP
from scripts.mni_constraints.shared43_study import NAME as ORIGINAL, MODELS, TARGETS
from scripts.mni_constraints.unetr_delayed_confirm import NAME as CONFIRM, PILOT, SEEDS

NAME='mni_final_recipes_constraints14_shared53t_0519_20261108'
MODULE='scripts.mni_constraints.final53_study'
NN_CONFIRM='mni_nnunet_repair_confirm53t_20261108'
NN_PILOT='mni_nnunet_repair_pilot53t_20261108'


def matrix():
    return [dict(id=f'MNI_{model}_{arm}_f{fold}s{seed}',kind='conventional',dataset='MNI',
        model=model,arm=arm,fold=fold,seed=seed,protocol=model+'_'+arm,
        augmentation='xy_d4_v1' if model in ('swinunetr','unetr_vit_b16') else 'none',
        train_subjects=10,target=TARGETS[model] if arm!='dice' else None)
        for arm in ARMS for seed in SEEDS for fold in range(3) for model in MODELS]


def reuse_origin(task):
    m,a,s,f=task['model'],task['arm'],task['seed'],task['fold']
    if m=='nnunet' and a=='dice':
        return ROOT/'reports'/(NN_PILOT if (s,f)==(19,0) else NN_CONFIRM)/'runs'/f'MNI_nnunet_lr002_f{f}s{s}'
    if m=='unetr_vit_b16' and a in ('dice','dice_bands'):
        if (s,f)==(19,0):
            return (ROOT/'reports'/PARENT/'runs/MNI_unetr_vit_b16_dice_f0s19' if a=='dice'
                    else ROOT/'reports'/PILOT/'runs/MNI_unetr_delayed_t05_f0s19')
        return ROOT/'reports'/CONFIRM/'runs'/f"MNI_unetr_{'baseline' if a=='dice' else 'delayed_t05'}_f{f}s{s}"
    if m in ('swinunetr','basic_unet16') and a=='dice' and (s,f)==(19,0):
        return ROOT/'reports'/PARENT/'runs'/task['id']
    return None


def changed_protocol(old,profile):
    p=deepcopy(old);m=p['model_identity'];a=p['requested_arm_order'][0];alias=p['trainer_alias']
    prior=p['models'][alias]['objectives'][a];costs=deepcopy(profile['costs'])
    for arm,cost in costs.items():cost['update_flops']+=int('bands' in arm)+int('edge' in arm)
    timing={k:prior[k] for k in ('prefix_updates','sdf_prefix_updates','constraint_warmup_updates')}
    if m=='unetr_vit_b16' and a!='dice':
        timing.update(prefix_updates=102,constraint_warmup_updates=33,sdf_prefix_updates=103 if 'sdf' in a else 102)
    s=schedule(costs,a,timing,20,CAP)
    horizon=200 if m=='nnunet' else s['max_steps']
    warmup=0 if m=='nnunet' else 24 if m=='unetr_vit_b16' else math.ceil(.05*horizon)
    p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),seeds=list(SEEDS),folds=[0,1,2],budget_flops=CAP,
        models={alias:dict(parameters=profile['parameters'],objectives={a:s})},
        constraint_warmup_updates=s['constraint_warmup_updates'],lr_horizon_updates=horizon,
        lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=horizon,warmup_updates=warmup,
                                decay_start_index=horizon//2 if p['lr_policy']=='hold_half' else warmup),
        validation_target_flops=sorted({CAP*i//20 for i in range(1,21)}),
        budget_origin='Shared53305109592450 optimization FLOPs; calibration accounted separately.',
        primary_metric='Final-budget macro Dice; nine equally weighted seed/fold cells per model/arm, paired against that model baseline.',
        decision_rule='Fixed four models,14 arms,seeds0/5/19,folds0/1/2. Retain every result; no score-dependent stopping,seed replacement or recipe changes.',
        timing_policy='UNETR: calibrate after102 updates and ramp non-SDF terms over33 updates in every constrained arm. Other models retain established prefixes/ramps. SDF separately activates nearest5/9 of the optimization cap.',
        calibration_strength_policy='Fresh training-only calibration on all20 crops; established targets and75% p95 guard; fixed weights afterward.',
        selection_caveat='UNETR bands timing selected and confirmed on development folds; extending timing to all combinations is explicitly user-requested and remains an experiment, not a proven benefit.')
    if m=='nnunet':p['initial_lrs']={alias:.02}
    return p


def prepare(_target=None):
    gate=read(ROOT/'reports'/CONFIRM/'confirmation_results.json')
    assert gate['status']=='complete' and len(gate['rows'])==9 and gate['all_nine_improved']
    assert all(r['gain_pp']>0 for r in gate['rows']), 'UNETR consistency condition not satisfied'
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT;original=ROOT/'experiments'/ORIGINAL
    if (stage/'manifest.json').exists():return verify(stage)
    for origin in (parent,original):
        for name,sha in read(origin/'manifest.json').items():assert digest(origin/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for module in ('unetr_constraint_pilot.py','unetr_delayed_confirm.py','final53_study.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/module,stage/'source/scripts/mni_constraints'/module)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    shutil.copy2(original/'manifest.json',stage/'original_manifest.json')
    write(stage/'confirmation_gate.json',gate)
    from scripts.mni_constraints.nnunet_repair import runtime_source
    source=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    legacy=ROOT/'experiments'/NN_CONFIRM/'source/scripts/cross_dataset_campaign/capped_core.py'
    assert legacy.read_text()==runtime_source(source.read_text())
    write(stage/'nnunet_reuse_equivalence.json',dict(original_sha256=digest(source),legacy_sha256=digest(legacy),
        scope='Repair runtime only asserts nnUNet, explicitly reassigns the existing CE coefficient1.0 and logs it. Default native Dice and CE coefficients are both1.0; new preflight verifies this. LR0.02,SGD and schedule match the nine repair runs.'))
    for model in MODELS:
        for arm in ARMS:
            key=model+'_'+arm;folder=stage/'protocols'/key
            shutil.copytree(original/'protocols'/key,folder)
            shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
            write(folder/'protocol.json',changed_protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json')))
    reused=[]
    for task in matrix():
        origin=reuse_origin(task)
        if origin is None:continue
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        shutil.copytree(origin,stage/'reused_runs'/task['id'])
        reused.append(dict(task=task,source_archive=str(origin),manifest_sha256=digest(origin/'archive_manifest.json')))
    assert len(reused)==29
    write(stage/'reused_runs.json',reused)
    plan=dict(name=NAME,tasks=[t for t in matrix() if reuse_origin(t) is None],num_runs=475,
        logical_runs=504,logical_cells=504,reused_runs=29,models=list(MODELS),conditions=list(ARMS),
        folds=[0,1,2],seeds=list(SEEDS),budget_flops=CAP,calibration_targets=TARGETS,gpu_workers=2,
        excluded_nodes=['gnode04'],augmentation='model_specific',
        conventional_python=read(parent/'plan.json')['conventional_python'],graceful_stop_seconds=20700,
        archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User requests automatically starting the complete established3fold3seed all-model baseline/constraint sweep if delayed UNETR improves in all9 matched pairs. Explicitly extend later UNETR calibration/onset to every constrained combination; preserve other winning recipes,nnUNetLR0.02,shared cap,two GPUs.',
        selection_caveat=gate['limitations'],gate='All nine archive-verified delayed UNETR gains positive.',
        recovery_policy='Preserve older paused-stage interrupted recovery artifacts; this new finite stage does not delete them.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Training core/objectives unchanged. Configuration-only LR0.02 for nnUNet,extended53T per-arm schedules,and UNETR prefix102/ramp33 across all constrained arms;29 verified exact/equivalent logical cells reused.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.common53_study',MODULE).replace('hippo-common53','hippo-final53'))
    manifest=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/PARENT/f.relative_to(stage))
            for f in stage.rglob('*') if f.is_file() and manifest.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/PARENT/'manifest.json')
    for folder in (stage/'protocols').iterdir():
        for f in folder.iterdir():
            if f.name not in ('protocol.json','parent_protocol.json'):
                donors[str(f.relative_to(stage))]=str(REMOTE/ORIGINAL/'protocols'/folder.name/f.name)
    for entry in reused:
        origin=Path(entry['source_archive']);remote_origin=REMOTE/origin.parts[-3]/'runs'/origin.name
        for f in (stage/'reused_runs'/entry['task']['id']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':
                donors[str(f.relative_to(stage))]=str(remote_origin/f.name)
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');gate=read(stage/'confirmation_gate.json')
    assert gate['status']=='complete' and gate['all_nine_improved'] and len(gate['rows'])==9
    assert plan['tasks']==[t for t in matrix() if reuse_origin(t) is None]
    assert len(plan['tasks'])==475 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    assert plan['seeds']==list(SEEDS) and plan['folds']==[0,1,2]
    parent=read(stage/'parent_manifest.json');original=read(stage/'original_manifest.json')
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for model in MODELS:
        for arm in ARMS:
            key=model+'_'+arm;folder=stage/'protocols'/key
            assert digest(folder/'parent_protocol.json')==original[f'protocols/{key}/protocol.json']
            p=read(folder/'protocol.json');profile=read(folder/'profile.json');alias=p['trainer_alias']
            assert p==changed_protocol(read(folder/'parent_protocol.json'),profile)
            s=p['models'][alias]['objectives'][arm]
            assert campaign.training_flops(s['max_steps'],s)<=CAP<campaign.training_flops(s['max_steps']+1,s)
            for name,field in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),
                               ('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[field]
            assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
            if model=='nnunet':assert p['initial_lrs'][alias]==.02 and p['lr_horizon_updates']==200
            if model=='unetr_vit_b16' and arm!='dice':
                assert s['prefix_updates']==102 and s['constraint_warmup_updates']==33
            if 'sdf' in arm:assert s['sdf_prefix_updates']>s['prefix_updates']
    reused=read(stage/'reused_runs.json');assert len(reused)==29
    assert {r['task']['id'] for r in reused}=={t['id'] for t in matrix() if reuse_origin(t) is not None}
    reference=read(stage/'nnunet_reuse_equivalence.json')
    assert reference['original_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
    for row in reused:
        task=row['task'];folder=stage/'reused_runs'/task['id']
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==row['manifest_sha256']
        cfg=read(folder/'config.json');p=read(stage/'protocols'/task['protocol']/'protocol.json')
        s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
        assert cfg['seed']==task['seed'] and cfg['fold']==task['fold'] and cfg['budget_flops']==CAP
        assert cfg['schedule']==s and cfg['max_steps']==s['max_steps']
        assert cfg['initial_lr']==p['initial_lrs'][p['trainer_alias']] and cfg['lr_horizon_updates']==p['lr_horizon_updates']
        split=read(stage/'protocols'/task['protocol']/'splits_10patients.json')[task['fold']]
        assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
        assert cfg['dataset_sha256']==p['pkl_sha256'] and cfg['augmentation']==p['augmentation']
        if task['model']=='nnunet':
            assert cfg['script_sha256']==reference['legacy_sha256'] and cfg['nnunet_ce_weight']==1.
        else:assert cfg['script_sha256']==reference['original_sha256']
    for seed in SEEDS:
        for fold in range(3):
            root=stage/'reused_runs'
            baseline=read(root/f'MNI_unetr_vit_b16_dice_f{fold}s{seed}'/'validation_latest.json')['summary']['macro_hard_dice']
            delayed=read(root/f'MNI_unetr_vit_b16_dice_bands_f{fold}s{seed}'/'validation_latest.json')['summary']['macro_hard_dice']
            assert delayed>baseline,'Automatic start condition failed'
            row=next(r for r in gate['rows'] if r['seed']==seed and r['fold']==fold)
            assert abs(100*(delayed-baseline)-row['gain_pp'])<1e-10
    return plan


def preflight(stage):
    verify(stage)
    from scripts.mni_constraints.verify import check_stage
    from scripts.unetr_subfields import cap_extension as batch,core
    from scripts.mni_constraints import objectives
    from types import SimpleNamespace
    import torch
    result=check_stage(stage)
    # Verify the actual repaired native optimizer and loss, including the
    # default CE coefficient used by the byte-identical general runtime.
    folder=stage/'protocols/nnunet_dice';p=read(folder/'protocol.json')
    batch.bind(SimpleNamespace(protocol_dir=folder,model='nnunet',objective='dice',fold=0))
    model=core.base.build_model('nnunet',read(folder/'nnunet_architecture.json')).to('meta')
    criterion,optimizer=core.base.build_loss_and_optimizer('nnunet',model,p)
    assert isinstance(optimizer,torch.optim.SGD) and optimizer.param_groups[0]['lr']==.02
    assert optimizer.param_groups[0]['momentum']==.95 and optimizer.param_groups[0]['nesterov']
    assert criterion.loss.weight_ce==criterion.loss.weight_dice==1.
    # Existing generic numerical checks use default calibration settings;
    # also check every combination at the actual model-specific targets.
    label=torch.zeros((1,1,64,64,64),dtype=torch.long)
    label[:,:,20:30,20:40,20:40]=1;label[:,:,30:36,20:40,20:40]=2;label[:,:,36:42,20:40,20:40]=3
    sdf=objectives.signed_distance_maps(label)
    import numpy as np
    depth=objectives.geometry_weights(label,np.eye(4),p['depth_geometry'])
    actual=[]
    for identity in MODELS:
        torch.manual_seed(731)
        native=criterion if identity=='nnunet' else None
        outputs=([torch.randn((1,4,n,n,n),requires_grad=True) for n in (64,32,16,8)]
                 if native else torch.randn((1,4,64,64,64),requires_grad=True))
        for arm in ARMS[1:]:
            row=objectives.calibration_case(outputs,label,arm,sdf,native,depth,identity=identity)
            weights,stats=objectives.select_weights([row],arm,TARGETS[identity],.75)
            assert all(s['weighted_median_ratio']<=TARGETS[identity]+1e-8 and s['weighted_p95_ratio']<=.7500001 for s in stats.values())
            value=objectives.training_loss(outputs,label,arm,weights,1.,sdf,native,depth,1.,identity=identity)
            assert torch.isfinite(value)
            actual.append(dict(model=identity,arm=arm,target=TARGETS[identity],statistics=stats))
    result.update(status='passed',manifest_sha256=digest(stage/'manifest.json'),
                  actual_calibration_checks=actual,nnunet_lr=.02,nnunet_ce_weight=1.,gate_all_nine_positive=True)
    write(stage/'preflight.json',result);return result


def main():
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
