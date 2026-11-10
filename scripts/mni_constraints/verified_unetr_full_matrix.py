"""All-model matrix gated on the completed, archive-verified UNETR confirmation.

Preparation does not bypass the nine-pair condition. This module is separate
from historical builders so completed and active experiment manifests stay fixed.
"""
from copy import deepcopy
from math import isclose
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE,ARMS
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,final53_study as legacy
from scripts.mni_constraints import unetr_plain_cosine_study as confirmation
from scripts.mni_constraints import verified_unetr_matrix_recipe as recipe_helper
from scripts.mni_constraints import verified_matrix_recovery as recovery
from scripts.mni_constraints import verified_matrix_runner as runner

ROOT,CAP=confirmation.ROOT,confirmation.CAP
NAME='mni_verified_unetr_cosine_constraints14_53t_0519_20261109'
MODULE='scripts.mni_constraints.verified_unetr_full_matrix'
RESULT_FILENAME='plain_cosine_results.json'
PROFILE_FILENAME='dropout0_original_constraints.json'
RUNTIME_FILENAME='unetr_simple_recipe_runtime.py'
ADD_MODULES=('verified_unetr_full_matrix.py','verified_unetr_matrix_recipe.py',
             'verified_matrix_recovery.py','verified_matrix_runner.py',
             'collect_verified_matrix.py','unetr_simple_recipe_study.py')


def matrix(recipe):
    tasks=legacy.matrix()
    for task in tasks:
        if task['model']=='unetr_vit_b16':
            task['augmentation']=recipe['augmentation']
            task['dropout_rate']=recipe['dropout_rate']
    assert len(tasks)==len({t['id'] for t in tasks})==504
    return tasks


def gate_from_report():
    report=ROOT/'reports'/confirmation.NAME
    result=read(report/RESULT_FILENAME)
    selected=recipe_helper.selected_recipe(result,confirmation_study=confirmation)
    audit=read(report/'final_cluster_audit.json')
    assert audit['status']=='complete' and audit['stage_state']['status']=='complete'
    assert not audit['queue'] and audit['broader_sweep_paused']
    assert {r['id'] for r in audit['runs']}=={r['id'] for r in result['archived_runs']}
    for row in result['archived_runs']:
        folder=report/'runs'/row['id']
        verify_archive(folder,read(folder/'archive_manifest.json'))
        assert read(folder/'cleanup.json')['status']=='complete'
    return result,selected,audit


def prepare(_target=None):
    result,selected,audit=gate_from_report()
    stage=ROOT/'experiments'/NAME
    if (stage/'manifest.json').exists():return verify(stage)
    parent=ROOT/'experiments'/confirmation.NAME
    original=ROOT/'experiments'/legacy.ORIGINAL
    common=ROOT/'experiments'/legacy.PARENT
    confirmation.verify(parent)
    for name,sha in read(original/'manifest.json').items():assert digest(original/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ADD_MODULES:
        target=stage/'source/scripts/mni_constraints'/name
        assert not target.exists(), f'Unexpected overwrite of inherited source: {name}'
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,target)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    shutil.copy2(original/'manifest.json',stage/'original_manifest.json')
    write(stage/'confirmation_gate.json',result)
    write(stage/'selected_recipe.json',selected)
    write(stage/'confirmation_cluster_audit.json',audit)
    shutil.copytree(parent/'profile_evidence',stage/'profile_evidence')
    for model in legacy.MODELS:
        for arm in ARMS:
            key=model+'_'+arm;folder=stage/'protocols'/key
            shutil.copytree(original/'protocols'/key,folder)
            shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
            shutil.copy2(folder/'profile.json',folder/'parent_profile.json')
            if model=='unetr_vit_b16' and selected['recipe']['dropout_rate']==0.:
                shutil.copy2(stage/'profile_evidence'/PROFILE_FILENAME,folder/'profile.json')
            p=recipe_helper.changed_protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),
                selected['recipe'],stage_name=NAME,profile_sha256=digest(folder/'profile.json'))
            write(folder/'protocol.json',p)
    from scripts.mni_constraints.nnunet_repair import runtime_source
    core=stage/recovery.CORE
    repaired=ROOT/'experiments'/legacy.NN_CONFIRM/recovery.CORE
    assert repaired.read_text()==runtime_source(core.read_text())
    write(stage/'nnunet_reuse_equivalence.json',dict(original_sha256=digest(core),
        legacy_sha256=digest(repaired),scope='NN repair only asserts identity, reassigns default CE weight1.0 and logs it; LR0.02, native DiceCE1:1, SGD and schedule match. Preflight checks actual coefficients.'))
    available={r['id']:r for r in result['archived_runs']}
    reused=[]
    for task in matrix(selected['recipe']):
        source_protocol=None
        if task['model']=='unetr_vit_b16':
            id=confirmation.identifier(selected['recipe']['key'],task['arm'],task['seed'],task['fold'])
            if id not in available:continue
            origin=ROOT/'reports'/confirmation.NAME/'runs'/id
            source_protocol=parent/'protocols'/(selected['recipe']['key']+'_'+task['arm'])
            recipe_helper.assert_same_training_protocol(read(source_protocol/'protocol.json'),
                read(stage/'protocols'/task['protocol']/'protocol.json'))
            shutil.copytree(source_protocol,stage/'reuse_protocols'/task['protocol'],dirs_exist_ok=True)
        else:
            origin=legacy.reuse_origin(task)
            if origin is None:continue
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        shutil.copytree(origin,stage/'reused_runs'/task['id'])
        reused.append(dict(task=task,source_archive=str(origin),manifest_sha256=digest(origin/'archive_manifest.json')))
    reused_ids={r['task']['id'] for r in reused}
    assert recovery.TASK_ID not in reused_ids
    write(stage/'reused_runs.json',reused)
    metadata=ROOT/'reports/mni_unetr_constraint_goal_20261108/paused_swin_recovery_metadata.json'
    (stage/'recovery').mkdir()
    shutil.copy2(metadata,stage/'recovery/metadata.json')
    shutil.copytree(common/'protocols/swinunetr_dice',stage/'recovery/protocol')
    recovery.inputs(stage,expected_origin=REMOTE/recovery.OLD_NAME)  # No checkpoint copy here.
    tasks=[t for t in matrix(selected['recipe']) if t['id'] not in reused_ids]
    plan=deepcopy(read(common/'plan.json'))
    plan.update(name=NAME,tasks=tasks,num_runs=len(tasks),logical_runs=504,logical_cells=504,
        reused_runs=len(reused),models=list(legacy.MODELS),conditions=list(ARMS),seeds=[0,5,19],folds=[0,1,2],
        budget_flops=CAP,gpu_workers=2,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='After a shared original-constraint recipe improves UNETR in all9 matched pairs, automatically run the complete4model14arm3fold3seed matrix, extending verified UNETR base/timing to every combination; NNLR0.02 and other recipes unchanged.',
        gate='All9 final-budget paired gains positive; all confirmation runs durably archived and cleaned.',
        selection_caveat='Development-validation selection. Compare each arm against its SAME base recipe. Benefits in combinations other than the confirmed arm remain experimental.',
        recovery_policy='Resume protected zero-step Swinseed19fold1 with byte-identical old protocol/core/data/config; retire original only after evaluated durable archive and cleanup of resumed run.',
        decision_rule='Run every remaining logical cell; retain failures and scores. No score-dependent seed replacement, per-pair tuning or early rejection in the full matrix.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        added_modules=list(ADD_MODULES),unchanged_training_core_sha256=digest(core),
        changes='Winning same-recipe UNETR base and delayed policy for all original combinations. Existing training runtime/core/objectives unchanged; model dispatch occurs once before training. Exact verified results reused; protected Swin recovery resumes rather than duplicates.'))
    (stage/'run.sbatch').write_text((common/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.common53_study',MODULE).replace('hippo-common53','hippo-verified53'))
    pm=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/confirmation.NAME/f.relative_to(stage))
        for f in stage.rglob('*') if f.is_file() and pm.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None)
    for entry in reused:
        origin=Path(entry['source_archive']);remote_origin=REMOTE/origin.parts[-3]/'runs'/origin.name
        for f in (stage/'reused_runs'/entry['task']['id']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':
                donors[str(f.relative_to(stage))]=str(remote_origin/f.name)
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage)
    return verify(stage)


def verify_reused(stage,entry):
    task=entry['task'];folder=stage/'reused_runs'/task['id']
    verify_archive(folder,read(folder/'archive_manifest.json'))
    assert digest(folder/'archive_manifest.json')==entry['manifest_sha256']
    assert read(folder/'cleanup.json')['status']=='complete'
    cfg=read(folder/'config.json');done=read(folder/'completion.json')
    p=read(stage/'protocols'/task['protocol']/'protocol.json')
    s=p['models'][p['trainer_alias']]['objectives'][task['arm']]
    assert (cfg['model_identity'],cfg['objective'],cfg['seed'],cfg['fold'])==(task['model'],task['arm'],task['seed'],task['fold'])
    assert cfg['budget_flops']==CAP and cfg['schedule']==s and cfg['max_steps']==s['max_steps']
    assert done['updates']==s['max_steps'] and done['discarded_attempts']==0
    assert done['consumed_training_flops']==campaign.training_flops(s['max_steps'],s)<=CAP
    for cfgkey,pkey in (('augmentation','augmentation'),('lr_horizon_updates','lr_horizon_updates'),
                       ('dataset_sha256','pkl_sha256'),('precision','precision')):
        assert cfg[cfgkey]==p[pkey]
    assert cfg['initial_lr']==p['initial_lrs'][p['trainer_alias']]
    source_fields={
        'objective_source_sha256':'scripts/compute_constraint_objectives.py',
        'shared_source_sha256':'scripts/train_compute_matched_baselines.py',
        'sdf_source_sha256':'thesis/new_constraints/signed_distance.py',
        'sdf_timing_sha256':'scripts/capped_depth_delay/schedule.py',
        'combined_objective_sha256':'scripts/capped_depth_delay/objectives.py',
        'depth_geometry_sha256':'scripts/depth_supervision/geometry.py',
    }
    if task['model']=='nnunet':source_fields['native_objective_sha256']='scripts/nnunet_multi_objective.py'
    for field,file in source_fields.items():assert cfg[field]==digest(stage/'source'/file),field
    runtime=read(folder/'runtime_recipe.json')
    assert runtime['objective_sha256']==digest(stage/'source/scripts/mni_constraints/objectives.py')
    split=read(stage/'protocols'/task['protocol']/'splits_10patients.json')[task['fold']]
    assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
    equivalent=read(stage/'nnunet_reuse_equivalence.json')
    if task['model']=='nnunet':
        assert cfg['script_sha256']==equivalent['legacy_sha256'] and cfg['nnunet_ce_weight']==1.
    else:assert cfg['script_sha256']==equivalent['original_sha256']
    if task['model']=='unetr_vit_b16':
        origin=stage/'reuse_protocols'/task['protocol']/'protocol.json'
        assert cfg['protocol_sha256']==digest(origin)
        recipe_helper.assert_same_training_protocol(read(origin),p)
        model=read(folder/'model_recipe.json')
        assert model['declared']==p['unetr_simplification']
        assert model['runtime_source_sha256']==digest(stage/'source/scripts/mni_constraints'/RUNTIME_FILENAME)
        if 'unetr_peak_lr' in p:
            assert model['peak_lr']==p['unetr_peak_lr']==cfg['initial_lr']
        if p.get('loss_recipe') in ('unetr_foreground_squared_dice','unetr_foreground_squared_dice_ce025',
                                   'unetr_foreground_squared_dice_ce050'):
            loss=read(folder/'base_loss_recipe.json')
            assert loss['recipe']==p['loss_recipe'] and loss['arguments']==p['loss_arguments']
            assert loss['adapter_sha256']==p['base_loss_adapter_sha256']
            assert loss['loss_source_sha256']==p['base_loss_source_sha256']
            assert loss['protocol_sha256']==digest(origin)
            coefficient=p['loss_arguments'].get('lambda_ce',0.)
            assert loss['dense_cross_entropy_in_base']==(coefficient>0.)
            if coefficient>0.:assert loss['dense_cross_entropy_weight']==coefficient
            assert model['loss_recipe']==p['loss_recipe']
            assert loss['original_auxiliary_source_sha256']==runtime['objective_sha256']
    return read(folder/'validation_latest.json')['summary']['macro_hard_dice']


def verify(stage):
    manifest=read(stage/'manifest.json')
    for name,sha in manifest.items():assert digest(stage/name)==sha,name
    gate=read(stage/'confirmation_gate.json')
    selected=recipe_helper.selected_recipe(gate,confirmation_study=confirmation)
    assert read(stage/'selected_recipe.json')==selected
    parent=read(stage/'parent_manifest.json');original=read(stage/'original_manifest.json')
    for name,sha in parent.items():
        if name.startswith(('source/','profile_evidence/')) and name!='source/source_manifest.json':assert manifest[name]==sha,name
    for model in legacy.MODELS:
        for arm in ARMS:
            folder=stage/'protocols'/(model+'_'+arm);p=read(folder/'protocol.json')
            assert digest(folder/'parent_protocol.json')==original[f'protocols/{model}_{arm}/protocol.json']
            expected=recipe_helper.changed_protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),
                selected['recipe'],stage_name=NAME,profile_sha256=digest(folder/'profile.json'))
            assert p==expected
            if model=='unetr_vit_b16':
                assert digest(folder/'profile.json')==digest(stage/'profile_evidence'/PROFILE_FILENAME)
            else:assert digest(folder/'profile.json')==original[f'protocols/{model}_{arm}/profile.json']
            s=p['models'][p['trainer_alias']]['objectives'][arm]
            assert campaign.training_flops(s['max_steps'],s)<=CAP<campaign.training_flops(s['max_steps']+1,s)
            assert p['pkl_sha256']==manifest['data/MNI.pkl']
            for file,key in (('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),
                             ('nnunet_architecture.json','nnunet_architecture_sha256')):
                assert digest(folder/file)==p[key]
    reused=read(stage/'reused_runs.json');ids={r['task']['id'] for r in reused}
    assert len(ids)==len(reused)
    scores={entry['task']['id']:verify_reused(stage,entry) for entry in reused}
    for row in gate['rows']:
        if row['candidate']!=selected['candidate']:continue
        stem=f"_f{row['fold']}s{row['seed']}"
        difference=100*(scores['MNI_unetr_vit_b16_'+selected['confirmed_arm']+stem]-scores['MNI_unetr_vit_b16_dice'+stem])
        assert difference>0 and abs(difference-row['gain_pp'])<1e-10
    tasks=matrix(selected['recipe']);plan=read(stage/'plan.json')
    assert ids<={t['id'] for t in tasks}
    assert plan['tasks']==[t for t in tasks if t['id'] not in ids]
    assert plan['num_runs']+plan['reused_runs']==plan['logical_cells']==504
    assert plan['num_runs']==len(plan['tasks']) and plan['reused_runs']==len(ids)
    assert plan['budget_flops']==CAP and plan['gpu_workers']==2
    assert plan['seeds']==[0,5,19] and plan['folds']==[0,1,2]
    recovery.inputs(stage,expected_origin=REMOTE/recovery.OLD_NAME)
    return plan


def check_runtime(stage):
    from contextlib import nullcontext
    context=getattr(recipe_helper.runtime,'use_base_loss',nullcontext)
    with context():
        return _check_runtime(stage)


def _check_runtime(stage):
    """Numerical/model checks only; this never prepares, submits or trains."""
    from scripts.mni_constraints.verify import check_stage
    from scripts.unetr_subfields import geo_recipe,core,cap_extension as batch
    from scripts.mni_constraints import objectives
    runtime=recipe_helper.runtime
    from types import SimpleNamespace
    import torch
    import numpy as np
    torch.set_num_threads(4)
    unetr=read(stage/'protocols/unetr_vit_b16_dice/protocol.json')
    original_builder=geo_recipe.build_model
    try:
        geo_recipe.build_model=lambda classes,shape:runtime.build_model(unetr)
        result=check_stage(stage)
    finally:geo_recipe.build_model=original_builder
    actual=[]
    label=torch.zeros((1,1,64,64,64),dtype=torch.long)
    # Longitudinal extent is along Y, not the identity affine's left-right X.
    label[:,:,24:40,16:29,26:38]=1
    label[:,:,24:40,29:36,26:38]=2
    label[:,:,24:40,36:48,26:38]=3
    for identity in legacy.MODELS:
        folder=stage/'protocols'/(identity+'_dice');p=read(folder/'protocol.json')
        args=SimpleNamespace(protocol_dir=folder,model=p['trainer_alias'],objective='dice',fold=0)
        trainer=runtime.bind(args) if identity=='unetr_vit_b16' else batch.bind(args)
        architecture=read(folder/'nnunet_architecture.json')
        if identity=='swinunetr':
            # MONAI reads linspace scalars during Swin construction; those
            # scalars need CPU storage before moving the model to meta.
            model=trainer.base.build_model(p['trainer_alias'],architecture).to('meta')
        else:
            with torch.device('meta'):
                model=trainer.base.build_model(p['trainer_alias'],architecture)
        criterion,optimizer=trainer.base.build_loss_and_optimizer(p['trainer_alias'],model,p)
        if identity=='nnunet':
            assert isinstance(optimizer,torch.optim.SGD) and optimizer.param_groups[0]['lr']==.02
            assert optimizer.param_groups[0]['momentum']==.95 and optimizer.param_groups[0]['nesterov']
            assert criterion.loss.weight_ce==criterion.loss.weight_dice==1.
        if identity=='unetr_vit_b16':
            trainer.check_no_augmentation(p)
            peak=p['initial_lrs']['unet']
            assert optimizer.param_groups[0]['lr']==peak
            assert isclose(trainer.base.learning_rate('unet',23,409,peak),peak,rel_tol=1e-12)
            if unetr['lr_policy']=='cap_matched':
                assert trainer.base.learning_rate('unet',205,409,peak)<peak*2/3
            else:
                assert unetr['lr_policy']=='hold_half'
                assert trainer.base.learning_rate('unet',203,409,peak)==peak
                assert trainer.base.learning_rate('unet',205,409,peak)>peak*59/60
            assert isclose(trainer.base.learning_rate('unet',408,409,peak),peak*.01,rel_tol=1e-12)
        native=criterion if identity=='nnunet' else None
        torch.manual_seed(731)
        outputs=([torch.randn((1,4,n,n,n),requires_grad=True) for n in (64,32,16,8)]
                 if native else torch.randn((1,4,64,64,64),requires_grad=True))
        sdf=objectives.signed_distance_maps(label)
        depth=objectives.geometry_weights(label,np.eye(4),p['depth_geometry'])
        for arm in ARMS[1:]:
            row=objectives.calibration_case(outputs,label,arm,sdf,native,depth,identity=identity)
            target=p['calibration_target_ratio']
            weights,stats=objectives.select_weights([row],arm,target,.75)
            assert all(v['weighted_median_ratio']<=target+1e-8 and v['weighted_p95_ratio']<=.7500001 for v in stats.values())
            loss=objectives.training_loss(outputs,label,arm,weights,1.,sdf,native,depth,1.,identity=identity)
            logits=outputs[0] if native else outputs
            assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,logits)[0]).all()
            actual.append(dict(model=identity,arm=arm,target=target,statistics=stats))
    result.update(status='passed',actual_calibration_checks=actual,nnunet_lr=.02,nnunet_ce_weight=1.,
        unetr_actual_dropout=unetr['dropout_rate'],unetr_lr_policy=unetr['lr_policy'],
        unetr_peak_lr=unetr['initial_lrs']['unet'],
        unetr_base_loss=unetr.get('loss_recipe','legacy_DiceCE'),unetr_base_loss_arguments=unetr.get('loss_arguments'),optimizer_updates=0)
    return result


def preflight(stage):
    verify(stage)
    result=check_runtime(stage)
    result.update(manifest_sha256=digest(stage/'manifest.json'),gate_all_nine_positive=True)
    write(stage/'preflight.json',result)
    return result


def main():
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=recipe_helper.train
    runner.install()
    campaign.main()


if __name__=='__main__':main()
