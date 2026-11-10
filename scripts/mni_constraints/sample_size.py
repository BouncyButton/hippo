"""Nested MNI subject-count pilots and a separately frozen matched comparison."""
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
MODULE='scripts.mni_constraints.sample_size'
PARENT='mni_adaptive_outer_bands_cap2_f0s0_20261107'
ORIGINAL_BANDS_PARENT='mni_shared_cap2_baseline_bands_f0s0_20261107'
MODELS=('swinunetr','unetr_vit_b16')
ALL_MODELS=('nnunet','swinunetr','unetr_vit_b16','basic_unet16')
CONDITIONS=('baseline','original_bands','adaptive')
COUNTS=(10,20)
REFERENCE_CAP=63952901334180
RUN_ONE=batch.run_one


def tasks(phase,reuse_pilots=False,counts=COUNTS,models=MODELS,conditions=CONDITIONS,band_targets=None):
    return [dict(id=f'MNI_{m}_n{n}_{v}_f0s0',kind='conventional',dataset='MNI',model=m,
                 arm='dice' if v=='baseline' else 'dice_bands',fold=0,seed=0,
                 protocol=f'{m}_n{n}_{v}',condition=v,train_subjects=n,
                 augmentation='xy_d4_v1' if m in MODELS else 'none',target=None if v=='baseline' else (band_targets or {}).get(m,bands.TARGETS[m]))
            for v in (('baseline',) if phase=='pilot' else tuple(v for v in conditions if not reuse_pilots or v!='baseline'))
            for n in counts for m in models]


def nested_splits():
    import numpy as np
    full=read(ROOT/'datasets/Dataset102_MNI/splits_final.json')[0]
    subject=lambda c:c.rsplit('_',1)[0]
    pool=sorted({subject(c) for c in full['train']})
    order=np.random.RandomState(0).permutation(pool).tolist()
    original=read(ROOT/'experiments'/PARENT/'protocols/swinunetr_baseline/splits_10patients.json')[0]
    assert set(order[:5])=={subject(c) for c in original['train']}
    assert sorted(full['val'])==original['val'] and len(pool)==20
    splits={n:dict(train=sorted(c for c in full['train'] if subject(c) in set(order[:n])),val=original['val']) for n in COUNTS}
    return splits,dict(selection='First N subjects from the original RandomState(0) permutation of fold0 training subjects; no validation-based selection.',
        subset_seed=0,subject_order=order,original_five=sorted(order[:5]),
        validation_subjects=sorted({subject(c) for c in original['val']}),
        source_split_sha256=digest(ROOT/'datasets/Dataset102_MNI/splits_final.json'))


def schedule(profile,arm,cap,count,timing=None):
    if arm=='dice':
        result=batch.schedule(profile['update_flops'],cap)
    elif timing is not None:
        from scripts.cross_dataset_campaign.prepare import schedule as measured_schedule
        costs=deepcopy(profile['costs']);costs['dice_bands']['update_flops']+=1
        result=measured_schedule(costs,arm,timing,count,cap)
    else:
        result=bands.band_schedule(profile,cap)
    result['patient_pass_epochs']=result['max_steps']/count
    return result


def prepare(name,cap=REFERENCE_CAP,phase='pilot',selection_path=None,reuse_stage=None,counts=COUNTS,models=MODELS,conditions=CONDITIONS,initial_lrs=None,paired_predictions=None,trajectory_stage=None,band_targets=None,matched_reuse=None):
    assert phase in ('pilot','comparison') and type(cap)==int and 0<cap<=2*REFERENCE_CAP
    counts=tuple(counts)
    assert counts and counts==tuple(sorted(set(counts))) and set(counts)<=set(COUNTS)
    models=tuple(models);conditions=tuple(conditions)
    assert models and len(models)==len(set(models)) and set(models)<=set(ALL_MODELS)
    assert conditions in (CONDITIONS,('baseline','original_bands'))
    assert not reuse_stage or conditions==CONDITIONS
    initial_lrs=dict(initial_lrs or {})
    band_targets=dict(band_targets or {})
    assert set(band_targets)<=set(models) and all(math.isfinite(v) and 0<v<=.6 for v in band_targets.values())
    assert set(initial_lrs)<=set(models) and all(math.isfinite(v) and v>0 for v in initial_lrs.values())
    assert not reuse_stage or not initial_lrs
    paired_predictions=not bool(reuse_stage) if paired_predictions is None else paired_predictions
    assert type(paired_predictions) is bool and not (reuse_stage and paired_predictions)
    stage=ROOT/'experiments'/name; parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists(): return verify(stage)
    trajectory={}
    if trajectory_stage:
        assert phase=='comparison' and not reuse_stage
        previous=ROOT/'experiments'/trajectory_stage;previous_plan=verify(previous)
        assert cap<=previous_plan['budget_flops']
        trajectory=dict(stage=trajectory_stage,manifest_sha256=digest(previous/'manifest.json'),protocols={})
        for task in tasks(phase,False,counts,models,conditions):
            trajectory['protocols'][task['protocol']]=read(previous/'protocols'/task['protocol']/'protocol.json')
    if phase=='comparison':
        assert selection_path is not None
        selection=read(selection_path)
        assert selection['status']=='selected' and selection['shared_cap']==cap
    splits,provenance=nested_splits()
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('sample_size.py','collect_sample_size.py','adaptive_diagnostics.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    shutil.copy2(ROOT/'scripts/cross_dataset_campaign/collect.py',stage/'source/scripts/cross_dataset_campaign/collect.py')
    trainer=stage/'source/scripts/cross_dataset_campaign/capped_core.py'
    source=trainer.read_text().replace('All ten training volumes','All training volumes')
    if phase=='pilot':
        marker="        from scripts.mni_constraints.adaptive_diagnostics import capture\n        capture(model, train, validation, device, amp, args, protocol, ledger['charged_attempts'], ledger['training_flops'], max_steps, native_criterion)\n"
        assert source.count(marker)==1; source=source.replace(marker,'')
    trainer.write_text(source)
    assert reuse_stage is None or phase=='comparison'
    references={}
    for task in tasks(phase,bool(reuse_stage),counts,models,conditions,band_targets):
        m,n,v,arm=task['model'],task['train_subjects'],task['condition'],task['arm']; count=2*n
        origin=(ROOT/'experiments'/ORIGINAL_BANDS_PARENT/'protocols'/f'{m}_dice_bands' if v=='original_bands'
                else parent/'protocols'/f'{m}_{v}')
        folder=stage/'protocols'/task['protocol']
        shutil.copytree(origin,folder); p=read(origin/'protocol.json');references[task['protocol']]=deepcopy(p)
        assert task['augmentation']==p['augmentation']
        write(folder/'splits_10patients.json',[splits[n]])
        if (folder/'split_provenance.json').exists(): (folder/'split_provenance.json').unlink()
        write(folder/'split_provenance.json',dict(**provenance,train_subjects=n,training_cases=splits[n]['train']))
        profile=read(folder/'profile.json')
        if arm!='dice':
            case_costs={r['composed_flops'] for r in profile['profiles'] if r['calibration'] and r.get('arm','dice_bands')=='dice_bands'}
            assert len(case_costs)==1
            profile['costs']['dice_bands']['calibration_flops']=count*case_costs.pop()+4096*(count//10)
            profile['calibration_cases']=count
            profile['calibration_accounting']='Measured per-case eval-forward and detached-logit gradient cost times actual training crops; historical4096 host selection allowance scaled with case count. Optimization cost unchanged.'
            write(folder/'profile.json',profile)
        alias=p['trainer_alias'];prior=trajectory.get('protocols',{}).get(task['protocol'])
        timing=prior['models'][alias]['objectives'][arm] if prior else None
        s=schedule(profile,arm,cap,count,timing); maximum=s['max_steps']
        if m in initial_lrs:p['initial_lrs']={alias:initial_lrs[m]}
        if arm!='dice' and m in band_targets:
            p['calibration_target_ratio']=band_targets[m]
            p['calibration_strength_policy']='Explicit requested median gradient target with unchanged75% p95 guard. Recalibrate on all training crops against the actual base loss; do not blindly multiply stored coefficients.'
        warmup=0 if m=='nnunet' else 24 if m=='unetr_vit_b16' else math.ceil(.05*maximum)
        p.update(name=name,data=str(REMOTE/name/'data/MNI.pkl'),train_count=count,validation_count=10,
            train_subjects=n,label_budget=f'{n} training subjects / {count} crops',folds=[0],seeds=[0],
            splits_sha256=digest(folder/'splits_10patients.json'),split_provenance_sha256=digest(folder/'split_provenance.json'),
            profile_sha256=digest(folder/'profile.json'),budget_flops=cap,
            models={alias:dict(parameters=profile['parameters'],objectives={arm:s})},
            lr_horizon_updates=maximum,constraint_warmup_updates=s['constraint_warmup_updates'],
            validation_target_flops=[cap*i//20 for i in range(1,21)],
            lr_schedule_details=dict(policy=p['lr_policy'],horizon_updates=maximum,warmup_updates=warmup,decay_start_index=maximum//2 if p['lr_policy']=='hold_half' else warmup),
            parent_stage=ORIGINAL_BANDS_PARENT if v=='original_bands' else PARENT,parent_protocol_sha256=digest(origin/'protocol.json'),
            adaptive_variant=v,adaptive_source_sha256=digest(stage/'source/scripts/mni_constraints/adaptive_bands.py'),
            budget_origin='One shared optimization cap for both models and subject counts; not scaled to preserve epochs.',
            training_dice_monitor=f'Unaugmented evaluation of all{count} training crops every5% optimization budget; validation remains the same10 crops from5 subjects.',
            calibration_policy='None: baseline only.' if arm=='dice' else 'Fresh calibration on every training crop; declared target and p95 guard; calibration FLOPs reported separately.',
            baseline_only=arm=='dice',diagnostic_cache_group=f'{m}_n{n}',paired_baseline_predictions=paired_predictions,
            spatial_diagnostics=('Standard train/validation Dice and confusion curves only for cap pilots.' if phase=='pilot' else
                'Spatial and training-only gradient probes at25/50/75/100%; aggregate baseline comparison; paired voxel transitions unavailable.' if not paired_predictions else
                'Spatial and training-only gradient probes at25/50/75/100%; paired comparison within the same subject count.'),
            constraint_recipe=('None.' if arm=='dice' else 'Original two-step union inner/outer bands with balanced grouped foreground/background BCE; existing gradient targets,prefix and ramp.' if v=='original_bands' else
                'Original union bands plus six-connected one-step halo around current external errors;3x detached error weights, GT-side normalization. Same expanded-band loss as five-subject pilot.'))
        if prior:
            for key in ('initial_lrs','lr_policy','augmentation','model_identity','training_loss','headroom_recipe','classes','spatial_size','splits_sha256','profile_sha256'):
                assert p.get(key)==prior.get(key),key
            p['lr_horizon_updates']=prior['lr_horizon_updates']
            p['lr_schedule_details']=deepcopy(prior['lr_schedule_details'])
        write(folder/'protocol.json',p)
    controls=[]
    for m in models:
        for v in conditions:
            archive=(ROOT/'reports'/ORIGINAL_BANDS_PARENT/'runs'/f'MNI_{m}_dice_bands_f0s0' if v=='original_bands'
                     else ROOT/'reports'/PARENT/'runs'/f'MNI_{m}_{v}_f0s0')
            verify_archive(archive,read(archive/'archive_manifest.json'));assert read(archive/'cleanup.json')['status']=='complete'
            folder=stage/'controls'/m/v;folder.mkdir(parents=True)
            for filename in ('config.json','metrics.csv','validation_latest.json','training_final.json','completion.json'):
                shutil.copy2(archive/filename,folder/filename)
            controls.append(dict(model=m,variant=v,archive_manifest_sha256=digest(archive/'archive_manifest.json')))
    pilot_controls=[]
    if reuse_stage:
        pilot=ROOT/'reports'/reuse_stage
        for relative,sha in read(pilot/'frozen/manifest.json').items():assert digest(pilot/'frozen'/relative)==sha
        assert read(pilot/'frozen/plan.json')['phase']=='pilot'
        assert read(pilot/'frozen/plan.json')['budget_flops']==cap
        for task in tasks('pilot',counts=counts,models=models):
            archive=pilot/'runs'/task['id'];verify_archive(archive,read(archive/'archive_manifest.json'))
            assert read(archive/'cleanup.json')['status']=='complete'
            config=read(archive/'config.json');m,n=task['model'],task['train_subjects']
            assert config['protocol_sha256']==digest(pilot/'frozen/protocols'/task['protocol']/'protocol.json')
            assert config['budget_flops']==cap and config['seed']==config['fold']==0
            assert config['train_cases']==splits[n]['train'] and config['validation_cases']==splits[n]['val']
            assert config['dataset_sha256']==digest(stage/'data/MNI.pkl') and config['augmentation']=='xy_d4_v1'
            previous=read(pilot/'frozen/protocols'/task['protocol']/'protocol.json')
            current=read(stage/'protocols'/f'{m}_n{n}_adaptive/protocol.json')
            for key in ('training_loss','initial_lrs','lr_policy','loss_recipe','headroom_recipe','spatial_size','classes','model_identity','trainer_alias'):
                assert previous.get(key)==current.get(key),key
            # Frozen parent recipes are identical; only cohort and shared cap were changed in the pilots.
            assert previous['parent_protocol_sha256']==digest(parent/'protocols'/f'{m}_baseline/protocol.json')
            folder=stage/'reused_baselines'/task['id'];shutil.copytree(archive,folder)
            pilot_controls.append(dict(task=task,archive=str(archive),archive_manifest_sha256=digest(archive/'archive_manifest.json'),
                protocol_sha256=digest(pilot/'frozen/protocols'/task['protocol']/'protocol.json')))
    matched=[]
    for model,reference_stage in (matched_reuse or {}).items():
        assert phase=='comparison' and model in ALL_MODELS and model not in models and not reuse_stage
        report=ROOT/'reports'/reference_stage;frozen=report/'frozen'
        for relative,sha in read(frozen/'manifest.json').items():assert digest(frozen/relative)==sha
        previous_plan=read(frozen/'plan.json');assert previous_plan['budget_flops']==cap
        for n in counts:
            for condition in conditions:
                task=next(t for t in previous_plan['tasks'] if t['model']==model and t['train_subjects']==n and t['condition']==condition and t['fold']==t['seed']==0)
                archive=report/'runs'/task['id'];verify_archive(archive,read(archive/'archive_manifest.json'))
                assert read(archive/'cleanup.json')['status']=='complete'
                config=read(archive/'config.json');protocol=frozen/'protocols'/task['protocol']
                assert config['budget_flops']==cap and config['fold']==config['seed']==0
                assert config['train_cases']==splits[n]['train'] and config['validation_cases']==splits[n]['val']
                assert config['dataset_sha256']==digest(stage/'data/MNI.pkl') and config['protocol_sha256']==digest(protocol/'protocol.json')
                runtime_files=('scripts/cross_dataset_campaign/capped_core.py','scripts/unetr_subfields/cap_extension.py','scripts/mni_constraints/objectives.py','scripts/mni_constraints/adaptive_diagnostics.py')
                for filename in runtime_files:assert digest(stage/'source'/filename)==digest(frozen/'source'/filename),filename
                shutil.copytree(archive,stage/'reused_comparisons'/task['id'])
                shutil.copytree(protocol,stage/'reused_protocols'/task['protocol'])
                matched.append(dict(task=task,source_stage=reference_stage,source_manifest_sha256=digest(frozen/'manifest.json'),
                    archive_manifest_sha256=digest(archive/'archive_manifest.json'),protocol_sha256=digest(protocol/'protocol.json')))
    plan=deepcopy(read(parent/'plan.json'))
    planned=tasks(phase,bool(reuse_stage),counts,models,conditions,band_targets)
    plan.update(name=name,phase=phase,tasks=planned,num_runs=len(planned),logical_runs=len(planned)+len(controls)+len(pilot_controls),reused_runs=len(controls)+len(pilot_controls),
        reuse_pilot_baselines=bool(reuse_stage),pilot_stage=reuse_stage,initial_lr_overrides=initial_lrs,band_target_overrides=band_targets,paired_baseline_predictions=paired_predictions,
        models=list(models),conditions=list(conditions),training_subject_counts=list(counts),folds=[0],seeds=[0],budget_flops=cap,
        archive_resume_module=MODULE,recipe_name='unchanged_recipes_nested_subject_count_shared_cap',
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/name)),
        authorization=('User requested extending the same 10-subject half-cap budget to nnUNet and UNet, baseline and original bands only, fold0 seed0, fixed validation, two GPU workers and per-run cleanup.' if models==('nnunet','basic_unet16') else 'User requested 10 training subjects and exploration of a lower shared cap, continuing SwinUNETR/UNETR baseline/original/expanded bands, fold0 seed0, fixed validation, two GPU workers and per-run cleanup.' if counts==(10,) else 'User requested10/20subjects,SwinUNETR/UNETR,baseline/expanded bands,fold0seed0,samevalidation,twoGPUworkers,per-run cleanup. Then explicitly requested short baseline pilots first to recalibrate one shared FLOP cap, and added original bands in the usual setup for both new subject counts.'),
        criterion=dict(primary='Pilot: assess training fit and persistent train/validation divergence across allfour cells before freezing one shared cap. Comparison: final-budget validation macro subfield Dice.',
            headroom='Flexible diagnostic, no rigid85–90% requirement and no model-specific caps. Do not multiply budget with subject count. A sustained late validation decline while training improves is an overfitting warning.',
            scope='Exploratory development comparison; cap frozen before this stage runs. Prior baseline and constrained outcomes may inform the next cap. No automatic additional folds/seeds.'),
        diagnostic_policy=('Verified identical-cap pilot baselines reused; no replay solely to recover retired prediction volumes. Expanded-band spatial diagnostics retained, but paired voxel transitions unavailable.' if reuse_stage else 'Pilots retain curves and confusion metrics, with no prediction caches. Comparison retains baseline prediction caches only until its last same-size constrained comparison is archived.'),
        limitation='This is a development-fold data-size comparison. Fixed compute gives fewer passes at larger N; lack of benefit can also reflect insufficient optimization. No cap can guarantee absence of overfitting.')
    for key in ('calibration_targets','multiplier'):plan.pop(key,None)
    if initial_lrs:
        plan['authorization']='User requested nnUNet initial learning rate0.01 at the same lower cap,10subjects,fold0seed0,baseline and original bands. All other optimization recipes retained; two GPU workers and per-run archive/cleanup.'
        plan['recipe_name']='explicit_learning_rate_override_same_subjects_and_cap'
    if not paired_predictions:
        plan['diagnostic_policy']='Each run independently retains Dice, confusion, spatial and training-gradient diagnostics. No paired prediction caches or baseline-completion dependency; both GPU workers can train concurrently.'
    if trajectory:
        plan['trajectory_stage']=trajectory_stage
        plan['authorization']='User requested setting nnUNet cap approximately at60% Dice, following the200-update maximum study. Stop at a fixed observed crossing while preserving its LR trajectory and constraint timing; same10subjects,fold0seed0,baseline/originalbands,twoGPU workers and per-run cleanup.'
        plan['recipe_name']='fixed_earlier_stop_preserving_observed_training_trajectory'
        write(stage/'trajectory_reference.json',trajectory)
    if band_targets:
        plan['authorization']='User requested historical6x stronger original bands under the current near60%-Dice nnUNet cap. Use60% median gradient target versus10%, retain75% p95 guard, same121updates,LR trajectory,20-update prefix/ramp,10subjects,fold0seed0. TwoGPU workers and per-run archive/cleanup.'
        plan['recipe_name']='stronger_calibrated_bands_same_fixed_trajectory_and_cap'
    if matched:
        plan['matched_reuse_count']=len(matched)
        plan['comparison_runs']=len(planned)+len(pilot_controls)+len(matched)
        plan['comparison_models']=[m for m in ALL_MODELS if m in set(models)|set(matched_reuse)]
        plan['logical_runs']+=len(matched);plan['reused_runs']+=len(matched)
        plan['authorization']='User requested all four MNI baselines and original bands at the shared43.01T optimization cap,10subjects,fold0seed0. Reuse verified matching nnUNet results at60% bands target with preserved trajectory; execute the other three model pairs using their established recipes and targets, schedules matched to cap, twoGPU workers and per-run cleanup.'
        plan['recipe_name']='four_model_shared_cap_original_bands_with_verified_completed_pairs'
        write(stage/'matched_reuse.json',matched)
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False));write(stage/'reference_protocols.json',references)
    write(stage/'nested_split_provenance.json',provenance);write(stage/'control_reuse.json',controls)
    write(stage/'pilot_baseline_reuse.json',pilot_controls)
    if selection_path:shutil.copy2(selection_path,stage/'cap_selection.json')
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        initial_lr_overrides=initial_lrs,trajectory_stage=trajectory_stage,band_target_overrides=band_targets,matched_reuse_stages=matched_reuse or {},
        changes='Training subjects and shared cap follow the plan; update LR horizon/pass metadata and calibration cohort arithmetic. Peak LR changes only via explicit initial_lr_overrides. Optional spatial caches scoped by model and subject count. Short pilots omit optional spatial/gradient probes; identical-cap pilot baselines may be reused with verified exact cohort/recipe/configuration and archive bindings. Baseline losses,augmentation,optimizers,architecture,validation and seed retained.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.adaptive_pilot',MODULE).replace('hippo-mni-adaptive','hippo-mni-samples'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');refs=read(stage/'reference_protocols.json');provenance=read(stage/'nested_split_provenance.json')
    trajectory=read(stage/'trajectory_reference.json') if plan.get('trajectory_stage') else {}
    counts=tuple(plan['training_subject_counts'])
    models=tuple(plan['models']);conditions=tuple(plan.get('conditions',CONDITIONS))
    assert counts and counts==tuple(sorted(set(counts))) and set(counts)<=set(COUNTS)
    assert models and len(models)==len(set(models)) and set(models)<=set(ALL_MODELS)
    assert conditions in (CONDITIONS,('baseline','original_bands'))
    targets=plan.get('band_target_overrides',{})
    assert set(targets)<=set(models) and all(math.isfinite(v) and 0<v<=.6 for v in targets.values())
    assert plan['tasks']==tasks(plan['phase'],plan.get('reuse_pilot_baselines',False),counts,models,conditions,targets) and plan['num_runs']==len(plan['tasks']) and plan['gpu_workers']==2
    allowed={'name','data','train_count','validation_count','train_subjects','label_budget','folds','seeds','splits_sha256','split_provenance_sha256',
        'profile_sha256','budget_flops','models','lr_horizon_updates','constraint_warmup_updates','validation_target_flops','lr_schedule_details',
        'parent_stage','parent_protocol_sha256','adaptive_variant','adaptive_source_sha256','budget_origin','training_dice_monitor','calibration_policy','baseline_only','diagnostic_cache_group','paired_baseline_predictions','spatial_diagnostics','constraint_recipe','initial_lrs','calibration_target_ratio','calibration_strength_policy'}
    expected_val=None;cohorts={}
    for task in plan['tasks']:
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json');old=refs[task['protocol']]
        assert {k for k in set(p)|set(old) if p.get(k)!=old.get(k)}<=allowed
        override=plan.get('initial_lr_overrides',{}).get(task['model'])
        if override is not None:assert math.isfinite(override) and override>0
        assert p['initial_lrs']==({p['trainer_alias']:override} if override is not None else old['initial_lrs'])
        assert p['calibration_target_ratio']==(targets.get(task['model'],old['calibration_target_ratio']) if task['arm']!='dice' else old['calibration_target_ratio'])
        if task['arm']!='dice':assert task['target']==p['calibration_target_ratio'] and p['calibration_p95_cap']==.75
        assert task['augmentation']==p['augmentation']
        n=task['train_subjects'];s=read(folder/'splits_10patients.json')[0]
        train_subjects={c.rsplit('_',1)[0] for c in s['train']};val_subjects={c.rsplit('_',1)[0] for c in s['val']}
        assert len(s['train'])==p['train_count']==2*n and len(s['val'])==10 and len(train_subjects)==n and len(val_subjects)==5
        assert train_subjects==set(provenance['subject_order'][:n]) and not train_subjects&val_subjects
        assert set(provenance['original_five'])<=train_subjects
        for person in train_subjects|val_subjects:assert sorted(c.rsplit('_',1)[1] for c in s['train']+s['val'] if c.rsplit('_',1)[0]==person)==['L','R']
        if expected_val is None:expected_val=s['val']
        assert s['val']==expected_val;cohorts[n]=train_subjects
        profile=read(folder/'profile.json');sched=p['models'][p['trainer_alias']]['objectives'][task['arm']]
        prior=trajectory.get('protocols',{}).get(task['protocol'])
        timing=prior['models'][p['trainer_alias']]['objectives'][task['arm']] if prior else None
        assert sched==schedule(profile,task['arm'],plan['budget_flops'],2*n,timing)
        assert p['lr_horizon_updates']==(prior['lr_horizon_updates'] if prior else sched['max_steps']) and p['budget_flops']==plan['budget_flops']
        if prior:
            assert trajectory['stage']==plan['trajectory_stage'] and sched['max_steps']<=timing['max_steps']
            for key in ('initial_lrs','lr_policy','lr_schedule_details','augmentation','model_identity','training_loss','headroom_recipe','classes','spatial_size','splits_sha256','profile_sha256'):
                assert p.get(key)==prior.get(key),key
        assert p.get('paired_baseline_predictions',True)==plan.get('paired_baseline_predictions',not plan.get('reuse_pilot_baselines',False))
        assert campaign.training_flops(sched['max_steps'],sched)<=p['budget_flops']<campaign.training_flops(sched['max_steps']+1,sched)
        if task['arm']!='dice':
            per_case={r['composed_flops'] for r in profile['profiles'] if r['calibration'] and r.get('arm','dice_bands')=='dice_bands'}
            assert len(per_case)==1 and sched['calibration_flops']==2*n*per_case.pop()+4096*(2*n//10)
        for filename,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('split_provenance.json','split_provenance_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/filename)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        assert p['adaptive_source_sha256']==digest(stage/'source/scripts/mni_constraints/adaptive_bands.py')
    for smaller,larger in zip(counts,counts[1:]):assert cohorts[smaller]<cohorts[larger]
    if plan['phase']=='comparison':assert read(stage/'cap_selection.json')['shared_cap']==plan['budget_flops']
    if plan.get('reuse_pilot_baselines'):
        reused=read(stage/'pilot_baseline_reuse.json');assert len(reused)==len(models)*len(counts)
        for item in reused:
            folder=stage/'reused_baselines'/item['task']['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
            assert digest(folder/'archive_manifest.json')==item['archive_manifest_sha256']
            config=read(folder/'config.json');assert config['budget_flops']==plan['budget_flops']
            current=read(stage/'protocols'/item['task']['protocol'].replace('_baseline','_adaptive')/'splits_10patients.json')[0]
            assert config['train_cases']==current['train'] and config['validation_cases']==current['val']
    if plan.get('matched_reuse_count'):
        items=read(stage/'matched_reuse.json');assert len(items)==plan['matched_reuse_count']
        seen=set()
        for item in items:
            task=item['task'];assert task['model'] not in models and task['fold']==task['seed']==0
            key=(task['model'],task['train_subjects'],task['condition']);assert key not in seen;seen.add(key)
            folder=stage/'reused_comparisons'/task['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
            assert read(folder/'cleanup.json')['status']=='complete' and digest(folder/'archive_manifest.json')==item['archive_manifest_sha256']
            config=read(folder/'config.json');protocol=stage/'reused_protocols'/task['protocol'];p=read(protocol/'protocol.json')
            assert config['protocol_sha256']==digest(protocol/'protocol.json')==item['protocol_sha256']
            assert config['budget_flops']==p['budget_flops']==plan['budget_flops'] and config['fold']==config['seed']==0
            assert config['dataset_sha256']==digest(stage/'data/MNI.pkl')
            s=read(protocol/'splits_10patients.json')[0]
            assert config['train_cases']==s['train'] and config['validation_cases']==s['val']==expected_val
            assert {c.rsplit('_',1)[0] for c in s['train']}==cohorts[task['train_subjects']]
            assert task['model']==p['model_identity'] and task['augmentation']==p['augmentation']
            if task['condition']!='baseline':assert task['target']==p['calibration_target_ratio']
        reused_models=set(plan['comparison_models'])-set(models)
        assert seen=={(m,n,c) for m in reused_models for n in counts for c in conditions}
    return plan


def preflight(stage):
    import torch
    from types import SimpleNamespace
    from scripts.unetr_subfields import core,baseline_runtime
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import adaptive_bands,objectives
    from scripts.mni_constraints.test_adaptive_bands import checks
    plan=verify(stage);torch.set_num_threads(4);scientific=checks();rows=[]
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    assert len(frame)==50 and frame.participant_id.nunique()==25
    for task in plan['tasks']:
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json');alias=p['trainer_alias'];arm=task['arm']
        batch.bind(SimpleNamespace(protocol_dir=folder,model=alias,objective=arm,fold=0))
        model=core.base.build_model(alias,read(folder/'nnunet_architecture.json')).to('meta')
        output=model(torch.zeros(1,1,64,64,64,device='meta'))
        shapes=[list(o.shape) for o in output] if isinstance(output,(list,tuple)) else [list(output.shape)]
        profile=read(folder/'profile.json')
        expected_shapes=profile.get('output_shapes')
        if expected_shapes is None:
            assert task['model']!='nnunet','Deep-supervision output shapes must be explicit'
            expected_shapes=[[1,4,64,64,64]]
        assert shapes==expected_shapes and shapes[0]==[1,4,64,64,64]
        assert sum(v.numel() for v in model.parameters())==p['models'][alias]['parameters']
        del model,output
        names=read(folder/'splits_10patients.json')[0]['train'];patients=core.PreparedSubfields(frame,names,DATASETS['MNI'])
        assert len(patients)==p['train_count']
        label=patients[0]['label'][None];torch.manual_seed(0)
        outputs=[torch.randn(shape,requires_grad=True) for shape in shapes]
        if task['model']!='nnunet':outputs=outputs[0]
        logits=outputs[0] if isinstance(outputs,list) else outputs
        criterion=None
        if task['model']=='nnunet':
            criterion,_=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER('nnunet',torch.nn.Linear(1,1),p)
        weights={}
        if arm!='dice':
            if task['condition']=='adaptive':adaptive_bands.install('adaptive')
            else:objectives.term=getattr(objectives.term,'_original_union_term',objectives.term)
            row=objectives.calibration_case(outputs,label,arm,criterion=criterion,identity=task['model'])
            weights,stats=objectives.select_weights([row],arm,p['calibration_target_ratio'],p['calibration_p95_cap'])
            assert weights['bands']>0 and stats['bands']['weighted_median_ratio']<=p['calibration_target_ratio']+1e-8
        value=objectives.training_loss(outputs,label,arm,weights,1.,criterion=criterion,identity=task['model']);g=torch.autograd.grad(value,logits)[0]
        assert torch.isfinite(value) and torch.isfinite(g).all()
        s=p['models'][alias]['objectives'][arm];maximum=s['max_steps'];lr=p['initial_lrs'][alias]
        horizon=p['lr_horizon_updates']
        assert batch.lr_at(horizon+10,horizon,lr,task['model'],p['lr_policy'])<=lr*.010001
        rows.append(dict(id=task['id'],train_subjects=task['train_subjects'],training_crops=len(patients),updates=maximum,passes=maximum/len(patients),calibration_flops=s['calibration_flops']))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=rows,scientific=scientific,optimizer_updates=0)
    write(stage/'preflight.json',result);return result


def train(args):
    from scripts.mni_constraints import adaptive_bands
    p=read(args.protocol_dir/'protocol.json')
    if p['adaptive_variant']=='adaptive':adaptive_bands.install('adaptive')
    bands.train(args)


def run_one(args):
    plan=read(args.stage/'plan.json');task=next(t for t in plan['tasks'] if t['id']==args.task)
    paired=plan.get('paired_baseline_predictions',not plan.get('reuse_pilot_baselines',False))
    if task['condition']!='baseline' and paired:
        baseline=args.stage/'runs'/f"MNI_{task['model']}_n{task['train_subjects']}_baseline_f0s0"
        while not (baseline/'cleanup.json').exists():
            if stopping():raise SystemExit(75)
            time.sleep(5)
        if task['condition']=='adaptive':
            original=args.stage/'runs'/f"MNI_{task['model']}_n{task['train_subjects']}_original_bands_f0s0"
            while not (original/'cleanup.json').exists():
                if stopping():raise SystemExit(75)
                time.sleep(5)
    RUN_ONE(args)
    last_condition=plan.get('conditions',CONDITIONS)[-1]
    if task['condition']==last_condition and task['condition']!='baseline' and paired:
        group=f"{task['model']}_n{task['train_subjects']}";receipt=args.stage/f'cache_cleanup_{group}.json'
        if receipt.exists():return
        cache=args.stage/'diagnostic_cache'/group;deleted=[]
        expected={r['name']:r for q in (25,50,75,100) for r in read(baseline/f'spatial_{q:03d}.json')['baseline_cache']}
        for path in sorted(cache.glob('*.npz')):
            info=expected[path.name];assert digest(path)==info['sha256'] and path.stat().st_size==info['bytes']
            deleted.append(info);path.unlink()
        assert not list(cache.iterdir());cache.rmdir()
        write(receipt,dict(status='complete',deleted=deleted,reason='Last matched constrained comparison durably archived; baseline predictions no longer needed.'))


def main():
    campaign.MODULE=MODULE;campaign.verify=verify;campaign.preflight=preflight;campaign.train=train;batch.run_one=run_one
    campaign.main()


if __name__=='__main__':main()
