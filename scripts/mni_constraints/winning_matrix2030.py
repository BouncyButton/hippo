"""Materialize the authorized matrix from frozen inputs and confirmed winners."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import time

from scripts.cross_dataset_campaign.common import ARMS, MODELS, REMOTE, read, write, digest
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, verified_lower_lr_matrix_recipe as recipe

NAME = 'mni_winning_recipes_matrix2030_53t_0519_20261109'
MODULE = 'scripts.mni_constraints.winning_matrix2030'
PARENT = 'mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
CAP = 53305109592450
START = 1794252600.0
PAIRS = {(s, f) for s in (0, 5, 19) for f in range(3)}


def promotion_score(model, rows):
    from statistics import mean
    if len(rows)!=9 or {(r['seed'],r['fold']) for r in rows}!=PAIRS:return None
    if not all(r['gain_pp']>0 for r in rows):return None
    if model=='swinunetr':
        class_consistent=all(r.get('ca4_gain_pp',float('-inf'))>0 for r in rows)
        macro_better=mean(r['vs_original_pp'] for r in rows)>0 and mean(r['ca4_vs_original_pp'] for r in rows)>=0
        class_better=class_consistent and mean(r['ca4_vs_original_pp'] for r in rows)>0
        if not (macro_better or class_better):return None
        # Predeclared before the seven new weak-bands confirmation pairs:
        # prioritize nine-pair CA4 consistency, with an explicit macro tradeoff.
        return [int(class_consistent),mean(r['ca4_vs_original_pp'] for r in rows),mean(r['vs_original_pp'] for r in rows),min(r['gain_pp'] for r in rows)]
    assert model=='nnunet'
    return [min(r['gain_pp'] for r in rows),mean(r['gain_pp'] for r in rows)]


def all_tasks():
    # Complete arms together, preserving matched baseline/constraint comparisons.
    return [dict(id=f'MNI_{m}_{a}_f{f}s{s}', kind='conventional', dataset='MNI',
                 model=m, arm=a, seed=s, fold=f, protocol=m+'_'+a,
                 augmentation='xy_d4_v1' if m == 'swinunetr' else 'none')
            for a in ARMS for s in (0, 5, 19) for f in range(3) for m in MODELS]


def check_bundle(bundle):
    for n, sha in read(bundle/'manifest.json').items():
        assert digest(bundle/n) == sha, n
    spec = read(bundle/'spec.json')
    assert spec['start_at'] == START and spec['budget_flops'] == CAP
    parent = Path(spec['parent'])
    assert parent == REMOTE/PARENT and digest(parent/'manifest.json') == spec['parent_manifest_sha256']
    assert read(parent/'control.json')['pause']
    return spec


def confirmed_candidates(stages):
    """Recompute every promotion gain from bound completed artifacts."""
    from statistics import mean
    found = {m: [] for m in ('nnunet', 'swinunetr')}
    for stage in stages:
        stage = Path(stage)
        assert stage.parent == REMOTE and stage.name.startswith('mni_nnunet_swin_refine2030_round')
        plan = read(stage/'plan.json')
        assert plan['deadline'] == START and plan['budget_flops'] == CAP
        for name, sha in read(stage/'manifest.json').items():
            assert digest(stage/name) == sha, name
        controls = {(c['model'], c['seed'], c['fold'], c['arm']): stage/'controls'/c['id']
                    for c in read(stage/'controls.json')}
        for model in found:
            keys = sorted({t['candidate'] for t in plan['tasks'] if t['model'] == model})
            for key in keys:
                tasks = [t for t in plan['tasks'] if t['model'] == model and t['candidate'] == key]
                assert {(t['seed'], t['fold']) for t in tasks} == PAIRS and len(tasks) == 9
                folders={t['id']:next((d for d in (stage/'runs'/t['id'],stage/'reused_pilot_runs'/t['id']) if (d/'completion.json').exists()),None) for t in tasks}
                if not all(folders.values()):
                    continue
                rows = []
                for t in tasks:
                    d = folders[t['id']];verify_archive(d, read(d/'archive_manifest.json'))
                    done = read(d/'completion.json');assert done['status'] == 'complete' and done['discarded_attempts'] == 0
                    cfg = read(d/'config.json');assert cfg['budget_flops'] == CAP
                    b = controls[model,t['seed'],t['fold'],'dice'];old = controls[model,t['seed'],t['fold'],'dice_bands']
                    for c in (b,old):verify_archive(c,read(c/'archive_manifest.json'))
                    v, bv, ov = [read(x/'validation_latest.json')['summary'] for x in (d,b,old)]
                    rows.append(dict(seed=t['seed'],fold=t['fold'],source=str(d),
                        archive_sha256=digest(d/'archive_manifest.json'),
                        gain_pp=100*(v['macro_hard_dice']-bv['macro_hard_dice']),
                        ca4_gain_pp=100*(v['CA4_DG_hard_dice']-bv['CA4_DG_hard_dice']),
                        vs_original_pp=100*(v['macro_hard_dice']-ov['macro_hard_dice']),
                        ca4_vs_original_pp=100*(v['CA4_DG_hard_dice']-ov['CA4_DG_hard_dice'])))
                score=promotion_score(model,rows)
                if score is None:continue
                found[model].append(dict(candidate=key,stage=str(stage),protocol=str(stage/'protocols'/tasks[0]['protocol']),
                    stage_manifest_sha256=digest(stage/'manifest.json'),rows=rows,score=score))
    return {m:max(v,key=lambda x:(x['score'],x['candidate'])) if v else None for m,v in found.items()}


def promote_protocol(old, profile, bands):
    """Extend the confirmed bands timing to bands-containing combinations only."""
    p = deepcopy(old);alias = p['trainer_alias'];arm = p['requested_arm_order'][0]
    if arm == 'dice_bands':return deepcopy(bands)
    assert 'bands' in arm
    prior = p['models'][alias]['objectives'][arm]
    chosen = bands['models'][alias]['objectives']['dice_bands']
    costs = deepcopy(profile['costs'])
    for a,cost in costs.items():cost['update_flops'] += int('bands' in a)+int('edge' in a)
    prefix = chosen['prefix_updates']
    # Preserve whether SDF has its distinct late activation under the original definition.
    timing = dict(prefix_updates=prefix, constraint_warmup_updates=chosen['constraint_warmup_updates'],
                  sdf_prefix_updates=prefix+int(prior['sdf_prefix_updates'] > prior['prefix_updates']))
    s = schedule(costs,arm,timing,20,CAP)
    p['models'][alias]['objectives'] = {arm:s}
    p.update(name=NAME,calibration_target_ratio=bands['calibration_target_ratio'],
        constraint_warmup_updates=chosen['constraint_warmup_updates'],
        timing_policy='Confirmed bands onset/ramp extended to this bands-containing combination; distinct SDF delay retained.',
        selection_caveat='Bands-only benefits confirmed on nine development pairs. This combination is a new experiment; its benefit is not assumed.')
    return p


def compatible(folder, protocol, source, task, equivalents, *, archive_already_verified=False):
    if not archive_already_verified:verify_archive(folder,read(folder/'archive_manifest.json'))
    cfg,done = read(folder/'config.json'),read(folder/'completion.json')
    alias=protocol['trainer_alias'];s=protocol['models'][alias]['objectives'][task['arm']]
    if (cfg['model_identity'],cfg['objective'],cfg['seed'],cfg['fold']) != (task['model'],task['arm'],task['seed'],task['fold']):return False
    for k,v in dict(schedule=s,max_steps=s['max_steps'],budget_flops=CAP,
        lr_horizon_updates=protocol['lr_horizon_updates'],initial_lr=protocol['initial_lrs'][alias],
        constraint_warmup_updates=protocol['constraint_warmup_updates'],
        precision=protocol['precision'],augmentation=protocol['augmentation'],dataset_sha256=protocol['pkl_sha256']).items():
        if cfg[k] != v:return False
    assert done['status']=='complete' and done['discarded_attempts']==0
    assert done['updates']==done['training_attempts_charged']==s['max_steps']
    assert done['consumed_training_flops']==campaign.training_flops(s['max_steps'],s)<=CAP
    assert cfg['script_sha256'] in ({equivalents['original_sha256'],equivalents['legacy_sha256']} if task['model']=='nnunet' else {equivalents['original_sha256']})
    assert read(folder/'runtime_recipe.json')['objective_sha256']==digest(source/'scripts/mni_constraints/objectives.py')
    split=read(source.parent/'protocols'/task['protocol']/'splits_10patients.json')[task['fold']]
    assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
    for field,name in (('objective_source_sha256','scripts/compute_constraint_objectives.py'),
                       ('shared_source_sha256','scripts/train_compute_matched_baselines.py'),
                       ('sdf_source_sha256','thesis/new_constraints/signed_distance.py'),
                       ('combined_objective_sha256','scripts/capped_depth_delay/objectives.py'),
                       ('depth_geometry_sha256','scripts/depth_supervision/geometry.py')):
        assert cfg[field]==digest(source/name),field
    if task['model']=='unetr_vit_b16':
        model=read(folder/'model_recipe.json')
        assert model['declared']==protocol['unetr_simplification']
        assert model['peak_lr']==protocol['initial_lrs'][alias]
    if task['arm']!='dice':
        c=read(folder/'calibration.json')
        if c['target_ratio']!=protocol['calibration_target_ratio'] or c['p95_cap']!=protocol['calibration_p95_cap']:return False
        if c['anchor_successful_updates']!=s['prefix_updates']:return False
    return True


def build(bundle, stage):
    spec=check_bundle(bundle);assert time.time()>=START
    if (stage/'manifest.json').exists():return verify(stage)
    parent=Path(spec['parent']);assert stage==REMOTE/NAME
    for n,sha in read(parent/'manifest.json').items():assert digest(parent/n)==sha,n
    winners=confirmed_candidates(read(bundle/'control.json')['refinement_stages'])
    stage.mkdir(exist_ok=True)
    assert not (stage/'status.json').exists() and not (stage/'submissions').exists()
    for part in ('source','protocols'):
        shutil.copytree(parent/part,stage/part,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (stage/'data').mkdir(exist_ok=True)
    if not (stage/'data/MNI.pkl').exists():os.link(parent/'data/MNI.pkl',stage/'data/MNI.pkl')
    assert digest(stage/'data/MNI.pkl')==digest(parent/'data/MNI.pkl')
    for name in ('winning_matrix2030.py','winning_matrix_runner.py'):
        shutil.copy2(bundle/name,stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    shutil.copy2(parent/'nnunet_reuse_equivalence.json',stage/'nnunet_reuse_equivalence.json')
    selected={'basic_unet16':spec['unet_confirmation'],**winners}
    bands_sources={}
    for model,item in selected.items():
        if item is None:continue
        origin=Path(item['protocol']);bands=read(origin/'protocol.json')
        if model=='basic_unet16':assert digest(origin/'protocol.json')==item['protocol_sha256']
        for arm in ARMS:
            if 'bands' not in arm:continue
            folder=stage/'protocols'/(model+'_'+arm)
            write(folder/'protocol.json',promote_protocol(read(folder/'protocol.json'),read(folder/'profile.json'),bands))
        for row in item['rows']:
            d=Path(row['source']);assert digest(d/'archive_manifest.json')==row['archive_sha256']
            bands_sources[model,row['seed'],row['fold']]=d
    equivalents=read(stage/'nnunet_reuse_equivalence.json');reused=[];pending=[]
    for task in all_tasks():
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        candidates=([bands_sources[task['model'],task['seed'],task['fold']]]
            if task['arm']=='dice_bands' and (task['model'],task['seed'],task['fold']) in bands_sources
            else [parent/'runs'/task['id'],parent/'reused_runs'/task['id']])
        origin=next((d for d in candidates if (d/'archive_manifest.json').exists() and compatible(d,p,stage/'source',task,equivalents)),None)
        if origin:
            shutil.copytree(origin,stage/'reused_runs'/task['id'],dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.pt','*.nii.gz'))
            reused.append(dict(task=task,source=str(origin),archive_sha256=digest(origin/'archive_manifest.json')))
        else:pending.append(task)
    # The interrupted UNETR SDF cell has no recipe changes and must resume exactly.
    rec=spec['recovery'];d=parent/'runs'/rec['id'];out=stage/'runs'/rec['id']
    assert any(t['id']==rec['id'] for t in pending)
    for n,sha in rec['files'].items():assert digest(d/n)==sha,n
    assert read(d/'config.json')['protocol_sha256']==digest(stage/'protocols/unetr_vit_b16_dice_sdf/protocol.json')
    shutil.copytree(d,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns('run.lock'))
    for n,sha in rec['files'].items():assert digest(out/n)==sha,n
    write(stage/'recovery.json',rec)
    write(stage/'selection.json',dict(selected_at=time.time(),winners=selected,unchanged_unetr=True,
        fallback_models=[m for m,w in winners.items() if w is None],extension_scope='bands-containing combinations only; UNETR existing all-combination recipe unchanged'))
    write(stage/'reused_runs.json',reused)
    plan=dict(name=NAME,tasks=pending,num_runs=len(pending),logical_cells=504,reused_runs=len(reused),
        models=list(MODELS),conditions=list(ARMS),seeds=[0,5,19],folds=[0,1,2],budget_flops=CAP,gpu_workers=2,
        conventional_python=spec['python'],graceful_stop_seconds=20700,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,archive=str(spec['local_report'])),
        authorization='Launch at20:30 with best confirmed shared model recipes,all3seeds/all3folds,original14conditions,twoGPUs.',
        decision_rule='Complete every remaining cell; no score-based rejection or seed replacement. Preserve exact interrupted recovery.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    (stage/'run.sbatch').write_text((bundle/'matrix_run.sbatch').read_text())
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),bundle_manifest_sha256=digest(bundle/'manifest.json'),training_core_unchanged=True))
    # Runtime runs/recovery files are mutable and must never enter the immutable manifest.
    write(stage/'source/source_manifest.json',{str(f.relative_to(stage/'source')):digest(f) for f in sorted((stage/'source').rglob('*')) if f.is_file() and f.name!='source_manifest.json' and '__pycache__' not in f.parts})
    write(stage/'manifest.json',{str(f.relative_to(stage)):digest(f) for f in sorted(stage.rglob('*'))
        if f.is_file() and 'runs' not in f.relative_to(stage).parts and f.name not in ('manifest.json','control.json') and '__pycache__' not in f.parts})
    return verify(stage)


def verify(stage):
    for n,sha in read(stage/'manifest.json').items():assert digest(stage/n)==sha,n
    plan=read(stage/'plan.json');assert plan['budget_flops']==CAP and plan['gpu_workers']==2
    reuse=read(stage/'reused_runs.json');ids={r['task']['id'] for r in reuse}
    assert len(ids)==len(reuse) and plan['tasks']==[t for t in all_tasks() if t['id'] not in ids]
    assert plan['num_runs']+plan['reused_runs']==504
    for n,sha in read(stage/'parent_manifest.json').items():
        if n.startswith('source/') and n!='source/source_manifest.json':assert digest(stage/n)==sha,n
    data_sha=digest(stage/'data/MNI.pkl')
    for t in {t['protocol']:t for t in all_tasks()}.values():
        folder=stage/'protocols'/t['protocol'];p=read(folder/'protocol.json');s=p['models'][p['trainer_alias']]['objectives'][t['arm']]
        assert p['constraint_warmup_updates']==s['constraint_warmup_updates']
        assert campaign.training_flops(s['max_steps'],s)<=CAP<campaign.training_flops(s['max_steps']+1,s)
        assert p['pkl_sha256']==data_sha
        for file,key in (('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')):assert digest(folder/file)==p[key]
    for entry in reuse:
        t=entry['task'];d=stage/'reused_runs'/t['id'];p=read(stage/'protocols'/t['protocol']/'protocol.json')
        assert digest(d/'archive_manifest.json')==entry['archive_sha256']
        # Every reused archive member is already bound by the stage manifest.
        am=read(d/'archive_manifest.json')
        frozen=read(stage/'manifest.json')
        for name,info in am['metadata'].items():assert frozen[str((d/name).relative_to(stage))]==info['sha256']
        assert compatible(d,p,stage/'source',t,read(stage/'nnunet_reuse_equivalence.json'),archive_already_verified=True)
    return plan


def preflight(stage):
    from scripts.mni_constraints import verified_lower_lr_full_matrix as inherited
    verify(stage)
    result=inherited.matrix.check_runtime(stage)
    # The inherited checks use each model's baseline target. Check any promoted
    # combination whose target differs using its actual native base loss too.
    import torch,numpy as np
    from scripts.mni_constraints import objectives as o
    from scripts.unetr_subfields import baseline_runtime
    label=torch.zeros((1,1,64,64,64),dtype=torch.long)
    label[:,:,24:40,16:29,26:38]=1;label[:,:,24:40,29:36,26:38]=2;label[:,:,24:40,36:48,26:38]=3
    sdf=o.signed_distance_maps(label);extra=[]
    for m in ('nnunet','swinunetr'):
        base=read(stage/'protocols'/(m+'_dice')/'protocol.json')
        criterion=baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER(m,torch.nn.Linear(1,1),base)[0] if m=='nnunet' else None
        z=([torch.randn((1,4,n,n,n),requires_grad=True) for n in (64,32,16,8)] if criterion else torch.randn((1,4,64,64,64),requires_grad=True))
        for arm in ARMS[1:]:
            p=read(stage/'protocols'/(m+'_'+arm)/'protocol.json');target=p['calibration_target_ratio']
            if target==base['calibration_target_ratio']:continue
            depth=o.geometry_weights(label,np.eye(4),p['depth_geometry'])
            row=o.calibration_case(z,label,arm,sdf,criterion,depth,identity=m)
            weights,stats=o.select_weights([row],arm,target,p['calibration_p95_cap'])
            loss=o.training_loss(z,label,arm,weights,1.,sdf,criterion,depth,1.,identity=m)
            assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,z[0] if isinstance(z,list) else z)[0]).all()
            assert all(s['weighted_p95_ratio']<=p['calibration_p95_cap']+1e-8 for s in stats.values())
            extra.append(dict(model=m,arm=arm,target=target))
    result['promoted_target_checks']=extra
    result.update(manifest_sha256=digest(stage/'manifest.json'))
    write(stage/'preflight.json',result)
    return result


def main():
    from scripts.mni_constraints import winning_matrix_runner as runner
    from scripts.unetr_subfields import cap_extension as batch
    campaign.MODULE=MODULE;campaign.verify=verify;campaign.preflight=preflight;campaign.train=recipe.train
    batch.run=runner.run
    campaign.main()


if __name__=='__main__':main()
