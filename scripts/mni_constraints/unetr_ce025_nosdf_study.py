"""Original bands/edge compositions without SDF under the unchanged quarter-CE base."""
from copy import deepcopy
from pathlib import Path
import shutil

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,seed_recovery as runner
from scripts.mni_constraints import unetr_delayed_combinations as source
from scripts.mni_constraints import unetr_ce025_runtime as runtime
from scripts.mni_constraints.unetr_recipe_pair_selection import select
from scripts.unetr_subfields import cap_extension as batch

ROOT,CAP,PAIRS,PILOTS=source.ROOT,source.CAP,source.PAIRS,source.PILOTS
NAME='mni_unetr_ce025_nosdf53t_20261109'
MODULE='scripts.mni_constraints.unetr_ce025_nosdf_study'
PARENT='mni_unetr_plain_ce02553t_20261109'
CORE='source/scripts/cross_dataset_campaign/capped_core.py'
RECIPES=(dict(key='plain_ce025',dropout_rate=0.,augmentation='none'),)
ARMS=('dice','dice_bands_edge_depth','dice_bands_edge')
LABELS={arm:arm.removeprefix('dice_') for arm in ARMS[1:]}
CANDIDATES=tuple(f"{r['key']}_{LABELS[a]}" for r in RECIPES for a in ARMS[1:])


def identifier(recipe,arm,seed,fold):
    return f'MNI_unetr_{recipe}_{arm}_f{fold}s{seed}'


def tasks():
    pairs=(*PILOTS,*(p for p in PAIRS if p not in PILOTS));result=[]
    for seed,fold in pairs:
        for recipe in RECIPES:
            for arm in ARMS:
                task=dict(id=identifier(recipe['key'],arm,seed,fold),recipe=recipe['key'],
                    role='baseline' if arm=='dice' else 'constraint',kind='conventional',dataset='MNI',
                    model='unetr_vit_b16',arm=arm,protocol=f"{recipe['key']}_{arm}",seed=seed,fold=fold,
                    augmentation=recipe['augmentation'],dropout_rate=recipe['dropout_rate'],pilot=(seed,fold) in PILOTS)
                if arm!='dice':task.update(candidate=f"{recipe['key']}_{LABELS[arm]}",
                    baseline_id=identifier(recipe['key'],'dice',seed,fold))
                result.append(task)
    return result


def protocol(old,profile,recipe):
    p=source.protocol(old,profile)
    p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),
        augmentation='none',dropout_rate=0.,unetr_simplification=dict(dropout_rate=0.,augmentation='none'),
        loss_recipe=runtime.LOSS_RECIPE,loss_arguments=runtime.dice.LOSS_ARGUMENTS,
        training_loss='Foreground-only squared softmax Dice plus denseCE weight0.25; original auxiliary losses unchanged.',
        base_loss_adapter_sha256=digest(Path(runtime.__file__)),
        base_loss_source_sha256=digest(Path(runtime.dice.__file__)),
        decision_rule='Exact matched Dice+0.25CE controls, reusing three verified baselines and training missing pairs; reject shared candidate after any nonpositive final pair; require9/9. Retain all attempts and compare absolute Dice against old winning and stronger plain DiceCE baselines.',
        timing_policy='Original102-update training-only calibration and33-update ramp. SDF omitted in both candidates; quarter-CE base, dropout0/noaug andhold-halfLR unchanged.',
        selection_caveat='Adaptive development-validation exploration on overlapping folds; positive gain over a weakened control is not superior absolute performance.',
        primary_metric='Final-budget constrained minus exact matched Dice+0.25CE baseline macro Dice.')
    assert p['lr_policy']=='hold_half'
    runtime.validate(p)
    assert p['lr_horizon_updates']==409
    return p


def same_training_protocol(left,right):
    descriptive={'name','data','primary_metric','timing_policy','selection_caveat','decision_rule'}
    assert left['pkl_sha256']==right['pkl_sha256']
    a={k:v for k,v in left.items() if k not in descriptive}
    b={k:v for k,v in right.items() if k not in descriptive}
    assert a==b, sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))


def prepare(_target=None):
    previous=read(ROOT/'reports'/PARENT/'plain_ce025_results.json')
    assert previous['status']=='complete' and previous['outcome']=='exhausted'
    decision=read(ROOT/'reports/mni_unetr_constraint_goal_20261108/ce025_nosdf_decision.json')
    assert decision['approved_for_execution'] and decision['recipes']==list(RECIPES) and decision['arms']==list(ARMS)
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    original=ROOT/'experiments'/source.ORIGINAL
    if (stage/'manifest.json').exists():return verify(stage)
    from scripts.mni_constraints import unetr_plain_ce025_study as previous_study
    previous_study.verify(parent)
    for part in ('source','data','controls'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('unetr_ce025_nosdf_study.py',):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    loss_source=stage/'source/scripts/unetr_subfields/foreground_dice_ce025.py'
    assert digest(loss_source)==digest(Path(runtime.dice.__file__))
    for name in ('controls.json','original_manifest.json'):
        shutil.copy2(parent/name,stage/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    write(stage/'previous_results.json',previous);write(stage/'decision.json',decision)
    profiles=ROOT/'reports/mni_unetr_constraint_goal_20261108/ce025_recipe_profiles'
    shutil.copytree(profiles,stage/'profile_evidence')
    assert read(profiles/'summary.json')['calibration_crops']==20
    assert digest(loss_source)==digest(Path(runtime.dice.__file__))
    for recipe in RECIPES:
        for arm in ARMS:
            folder=stage/'protocols'/f"{recipe['key']}_{arm}"
            shutil.copytree(original/'protocols'/('unetr_vit_b16_'+arm),folder)
            shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
            shutil.copy2(folder/'profile.json',folder/'parent_profile.json')
            if recipe['dropout_rate']==0.:
                shutil.copy2(profiles/'ce025_original_constraints.json',folder/'profile.json')
            p=protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),recipe)
            p['profile_sha256']=digest(folder/'profile.json')
            if arm=='dice':
                original_base=read(parent/'protocols/plain_ce025_dice/protocol.json')
                same_training_protocol(original_base,p)
                p=original_base
            write(folder/'protocol.json',p)
    references=[]
    from scripts.mni_constraints.analyze_unetr_delayed_confirm import folder_for
    for seed,fold in PAIRS:
        for arm in ARMS[1:]:
            origin=(folder_for(seed,fold,'delayed_t05') if arm=='dice_bands' else
                ROOT/'reports'/source.NAME/'runs'/f'MNI_unetr_{arm}_f{fold}s{seed}')
            if not (origin/'cleanup.json').exists():continue
            verify_archive(origin,read(origin/'archive_manifest.json'))
            name=f'{arm}_f{fold}s{seed}';shutil.copytree(origin,stage/'old_constraints'/name)
            references.append(dict(name=name,seed=seed,fold=fold,arm=arm,origin=str(origin),
                manifest_sha256=digest(origin/'archive_manifest.json')))
    write(stage/'old_constraints.json',references)
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=27,maximum_new_runs=24,pilot_runs=12,reused_baselines=3,
        recipes=list(RECIPES),conditions=list(CANDIDATES),archive_resume_module=MODULE,
        augmentation='recipe_specific',
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User explicitly authorizes simpler UNETR training after current constraint candidates fail. Dice+0.25CE base with fresh SAME-loss controls. TwoGPUworkers,original definitions,seeds0/5/19,all3folds,same53.305Tcap. Othermodels paused.',
        selection_rule='Two original-constraint compositions without SDF under the unchanged quarter-CE base. Reuse only the three exact matched controls already archived; train missing controls. Share controls only within identical base/seed/fold. Screen candidates with fewest completed pairs first. First4fixedpilots before5remainingpairs; reject afteranynonpositivegain; all9required. Retainalready-startedruns.',
        recipe='Unchanged Dice+0.25CE base crossed with original delayed5% bands+edge+depth and bands+edge; no SDF term. SameAdamW6e-4,24warmup,hold-halfLR,409updates. Training-only20cropcalibration at102,33-update ramp; no SDF term.',
        evidence=decision['reason'],scope='Controlled base-recipe interaction study; compare both matched constraint gain and absolute Dice against current winning baseline. No per-pair hyperparameter tuning.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        unchanged_training_core_sha256=digest(stage/CORE),
        changes='Training core and original auxiliary losses unchanged. Quarter-CE runtime and measured profiles preserved byte-for-byte. Test original bands+edge with/without depth; reuse three exact archived baseline protocols and train missing controls. All27 schedules have409updates.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.unetr_plain_ce025_study',MODULE).replace('hippo-unetr-ce025','hippo-unetr-ce025-nosdf'))
    pm=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/PARENT/f.relative_to(stage)) for f in stage.rglob('*')
            if f.is_file() and pm.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/PARENT/'manifest.json')
    for recipe in RECIPES:
        for arm in ARMS:
            folder=stage/'protocols'/f"{recipe['key']}_{arm}"
            for f in folder.iterdir():
                if f.name in ('splits_10patients.json','nnunet_architecture.json'):
                    donors[str(f.relative_to(stage))]=str(REMOTE/source.ORIGINAL/'protocols'/('unetr_vit_b16_'+arm)/f.name)
    for reference in references:
        remote_origin=REMOTE/reference['origin'].split('/reports/',1)[1]
        for f in (stage/'old_constraints'/reference['name']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':donors[str(f.relative_to(stage))]=str(remote_origin/f.name)
    reused=[]
    baselines=[r for r in previous['archived_runs'] if r['role']=='baseline']
    assert {(r['seed'],r['fold']) for r in baselines}==set(PILOTS[:3])
    for row in baselines:
        origin=ROOT/'reports'/PARENT/'runs'/row['id']
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        cfg=read(origin/'config.json')
        assert cfg['protocol_sha256']==digest(stage/'protocols/plain_ce025_dice/protocol.json')
        assert cfg['script_sha256']==digest(stage/CORE)
        shutil.copytree(origin,stage/'runs'/row['id'])
        shutil.copytree(origin,ROOT/'reports'/NAME/'runs'/row['id'])
        reused.append(dict(id=row['id'],origin=str(origin),manifest_sha256=digest(origin/'archive_manifest.json')))
        for f in (stage/'runs'/row['id']).iterdir():
            if f.is_file():donors[str(f.relative_to(stage))]=str(REMOTE/PARENT/'runs'/row['id']/f.name)
    write(stage/'reused_baselines.json',reused)
    write(stage/'deployment_known_donors.json',donors);campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    manifest=read(stage/'manifest.json')
    for name,sha in manifest.items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['num_runs']==27 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    assert read(stage/'previous_results.json')['outcome']=='exhausted'
    parent=read(stage/'parent_manifest.json');original=read(stage/'original_manifest.json')
    reused=read(stage/'reused_baselines.json');assert len(reused)==3
    for item in reused:
        folder=stage/'runs'/item['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==item['manifest_sha256']
        assert read(folder/'cleanup.json')['status']=='complete'
        assert read(folder/'config.json')['protocol_sha256']==digest(stage/'protocols/plain_ce025_dice/protocol.json')
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert manifest[name]==sha,name
    profile=read(stage/'profile_evidence/ce025_original_constraints.json')
    for name,sha in profile['source_sha256'].items():assert digest(stage/'profile_evidence/source'/name)==sha,name
    assert profile['base_loss_adapter_sha256']==digest(stage/'source/scripts/mni_constraints/unetr_ce025_runtime.py')
    assert profile['base_loss_source_sha256']==digest(stage/'source/scripts/unetr_subfields/foreground_dice_ce025.py')
    assert profile['objective_sha256']==digest(stage/'source/scripts/mni_constraints/objectives.py')
    for recipe in RECIPES:
        for arm in ARMS:
            tag=f"{recipe['key']}_{arm}";folder=stage/'protocols'/tag;p=read(folder/'protocol.json')
            assert manifest[f'protocols/{tag}/parent_protocol.json']==original[f'protocols/unetr_vit_b16_{arm}/protocol.json']
            assert manifest[f'protocols/{tag}/parent_profile.json']==original[f'protocols/unetr_vit_b16_{arm}/profile.json']
            expected=protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),recipe)
            expected['profile_sha256']=digest(folder/'profile.json')
            if arm=='dice':
                assert digest(folder/'protocol.json')==parent['protocols/plain_ce025_dice/protocol.json']
                same_training_protocol(p,expected)
            else:assert p==expected
            assert read(folder/'profile.json')==(profile if recipe['dropout_rate']==0. else read(folder/'parent_profile.json'))
            s=p['models']['unet']['objectives'][arm]
            assert s['max_steps']==p['lr_horizon_updates']==409
            assert campaign.training_flops(409,s)<=CAP<campaign.training_flops(410,s)
            assert p['lr_policy']=='hold_half' and p['lr_schedule_details']['decay_start_index']==204
            runtime.validate(p)
            assert p['calibration_target_ratio']==.05 and p['calibration_p95_cap']==.75
            assert p['pkl_sha256']==manifest['data/MNI.pkl']
            for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
                assert manifest[f'protocols/{tag}/{name}']==p[key]
    controls=read(stage/'controls.json');assert {(c['seed'],c['fold']) for c in controls}==set(PAIRS)
    for c in controls:
        folder=stage/'controls'/c['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==c['manifest_sha256']
    for item in read(stage/'old_constraints.json'):
        folder=stage/'old_constraints'/item['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==item['manifest_sha256']
    return plan


def selection(stage,plan):
    completed={}
    for task in plan['tasks']:
        folder=stage/'runs'/task['id']
        if (folder/'cleanup.json').exists():
            completed[task['id']]=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
    return select(plan['tasks'],completed,CANDIDATES)


def eligible(stage,plan,state):return state['pending']


def preflight(stage):
    verify(stage)
    import torch,monai
    from types import SimpleNamespace
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0';checks=[]
    for recipe in RECIPES:
        folder=stage/'protocols'/f"{recipe['key']}_dice";p=read(folder/'protocol.json')
        trainer=runtime.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective='dice',fold=0))
        trainer.check_no_augmentation(p)
        assert trainer.base.learning_rate('unet',23,409,.0006)==.0006
        assert trainer.base.learning_rate('unet',25,409,.0006)==.0006
        assert .00059<trainer.base.learning_rate('unet',205,409,.0006)<.0006
        assert abs(trainer.base.learning_rate('unet',408,409,.0006)-.000006)<1e-12
        with torch.device('meta'):
            model=trainer.base.build_model('unet',read(folder/'nnunet_architecture.json'))
            result=model(torch.zeros((1,1,64,64,64)))
        assert list(result.shape)==[1,4,64,64,64]
        checks.append(dict(recipe=recipe['key'],actual_dropout=recipe['dropout_rate'],augmentation=recipe['augmentation'],parameters=sum(x.numel() for x in model.parameters())))
    with runtime.use_base_loss():
        frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
        for fold,split in enumerate(read(folder/'splits_10patients.json')):
            data=core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
            assert len(data)==20 and len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
            item=data[0];label=item['label'][None];torch.manual_seed(811);z=torch.randn((1,4,64,64,64),requires_grad=True)
            sdf=objectives.signed_distance_maps(label)
            depth=objectives.geometry_weights(label,p['native_geometry'][item['case_name']]['model_affine'],p['depth_geometry'])
            for arm in ARMS[1:]:
                row=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
                weights,stats=objectives.select_weights([row],arm,.05,.75)
                assert all(v['weighted_median_ratio']<=.0500001 and v['weighted_p95_ratio']<=.7500001 for v in stats.values())
                loss=objectives.training_loss(z,label,arm,weights,1.,sdf,None,depth,1.,identity='unetr_vit_b16')
                assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,z)[0]).all()
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,
        model_checks=checks,all27schedules409updates=True,matched_controls_required=True,reused_baselines=3,original_losses=True)
    write(stage/'preflight.json',result);return result


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=()
    runner.verify,runner.selection,runner.eligible=verify,selection,eligible
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=runtime.train;batch.run=runner.run
    campaign.main()


if __name__=='__main__':main()
