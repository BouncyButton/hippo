"""Finite original-constraint windows with sequential nine-pair confirmation."""
from copy import deepcopy
import difflib
import shutil

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed_recovery as runner
from scripts.mni_constraints import unetr_delayed_combinations as source
from scripts.mni_constraints import unetr_constraint_window as window
from scripts.mni_constraints.unetr_sequential_selection import select
from scripts.unetr_subfields import cap_extension as batch

ROOT, CAP, PAIRS, PILOTS = source.ROOT, source.CAP, source.PAIRS, source.PILOTS
NAME = 'mni_unetr_window53t_20261108'
MODULE = 'scripts.mni_constraints.unetr_window_study'
PARENT = 'mni_unetr_adam_probe53t_20261108'
DIAGNOSTIC_PARENT = PARENT
PRIOR_RESULT = 'optimizer_probe_results.json'
PRIOR_OUTCOME = 'diagnostic_replay_verified'
DECISION_FILE = 'window_decision.json'
EXPECT_ZERO_FINAL = True
CORE = 'source/scripts/cross_dataset_campaign/capped_core.py'
CANDIDATES = tuple(dict(key=f'window{stop}_{label}', taper_start=start, stop=stop, arm=arm)
    for start,stop in ((205,307),(307,369))
    for label,arm in (('bands','dice_bands'),('full','dice_bands_edge_sdf_depth')))
KEYS = tuple(c['key'] for c in CANDIDATES)


def tasks():
    pairs = (*PILOTS, *(p for p in PAIRS if p not in PILOTS))
    return [dict(id=f"MNI_unetr_{c['key']}_f{fold}s{seed}", candidate=c['key'],
        **({k:c[k] for k in ('target',) if k in c}),
        protocol=c['key'], arm=c['arm'], taper_start=c['taper_start'], stop=c['stop'],
        seed=seed, fold=fold, pilot=(seed,fold) in PILOTS, kind='conventional',
        dataset='MNI', model='unetr_vit_b16', augmentation='xy_d4_v1')
        for seed,fold in pairs for c in CANDIDATES]


def reference_folder(task):
    if task.get('target',.05)!=.05:return None
    if task['arm']=='dice_bands':
        from scripts.mni_constraints.analyze_unetr_delayed_confirm import folder_for
        return folder_for(task['seed'],task['fold'],'delayed_t05')
    folder=ROOT/'reports'/source.NAME/'runs'/f"MNI_unetr_{task['arm']}_f{task['fold']}s{task['seed']}"
    return folder if (folder/'cleanup.json').exists() else None


def protocol(old, profile, candidate):
    p = source.protocol(old,profile); arm = candidate['arm']
    schedule = p['models']['unet']['objectives'][arm]
    schedule = window.account(schedule, 'sdf' in arm, CAP)
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'),
        calibration_target_ratio=candidate.get('target',.05),
        models={'unet':dict(parameters=profile['parameters'],objectives={arm:schedule})},
        auxiliary_envelope=window.envelope(409,candidate['taper_start'],candidate['stop']),
        finite_window=candidate,
        decision_rule='Fixed shared envelopes, same original losses and calibration. Reject after any nonpositive matched final gain, retaining all attempts. Four fixed development pilots then five further pairs; success requires all9.',
        timing_policy='Original102-update onset and33-update ramp, SDF nearest5/9 cap. Multiply all auxiliary terms by a shared predeclared linear fade to zero; finish with base-loss gradients. Original terms remain evaluated and charged, including zero-weight phase.',
        selection_caveat='Repeated development-validation exploration on overlapping folds, not an untouched-test claim.')
    return p


def prepare(_target=None):
    gate = read(ROOT/'reports'/PARENT/PRIOR_RESULT)
    assert gate['status']=='complete' and gate['outcome']==PRIOR_OUTCOME
    prior = read(ROOT/'reports'/DIAGNOSTIC_PARENT/'optimizer_probe_results.json')
    assert prior['status']=='complete' and prior['outcome']=='diagnostic_replay_verified'
    decision = read(ROOT/'reports/mni_unetr_constraint_goal_20261108'/DECISION_FILE)
    assert decision['approved_for_execution'] and decision['candidates']==list(CANDIDATES)
    stage=ROOT/'experiments'/NAME; parent=ROOT/'experiments'/source.NAME
    original=ROOT/'experiments'/source.ORIGINAL
    if (stage/'manifest.json').exists(): return verify(stage)
    source.verify(parent)
    for part in ('source','data','controls'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in sorted({'unetr_window_study.py','unetr_constraint_window.py','unetr_sequential_selection.py',MODULE.rsplit('.',1)[-1]+'.py'}):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    for name in ('controls.json','original_manifest.json'):
        shutil.copy2(parent/name,stage/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    reference=stage/'reference_source/capped_core.py';reference.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(stage/CORE,reference);before=reference.read_text();after=window.instrument_source(before)
    (stage/CORE).write_text(after)
    (stage/'window.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='reference/capped_core.py',tofile=CORE)))
    write(stage/'optimizer_evidence.json',prior);write(stage/'decision.json',decision)
    write(stage/'prior_stage_results.json',gate)
    references=[]
    for task in tasks():
        origin=reference_folder(task)
        name=f"{task['arm']}_f{task['fold']}s{task['seed']}"
        if origin is None or any(r['name']==name for r in references): continue
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        shutil.copytree(origin,stage/'persistent_references'/name)
        references.append(dict(name=name,seed=task['seed'],fold=task['fold'],arm=task['arm'],
            origin=str(origin),manifest_sha256=digest(origin/'archive_manifest.json')))
    write(stage/'persistent_references.json',references)
    for c in CANDIDATES:
        folder=stage/'protocols'/c['key']
        shutil.copytree(original/'protocols'/('unetr_vit_b16_'+c['arm']),folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        write(folder/'protocol.json',protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),c))
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=36,maximum_new_runs=36,pilot_runs=16,
        candidates=list(CANDIDATES),conditions=list(KEYS),archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='Continue explicit goal until one original-constraint UNETR recipe improves across seeds0/5/19 and folds0/1/2. Other models paused. Two GPU workers; same53.305T cap and winning base recipe.',
        evidence=decision['reason'],selection_rule='Sequential rejection after any nonpositive final pair; no future pairs for rejected candidates. Fixed four pilots before remaining five; first9/9 success ends stage. Retain concurrent already-started cells and every failure.',
        recipe='Original bands or original full combination,target.05,p95.75,prefix102,ramp33; fixed linear auxiliary taper to zero, with scalar operation accounting.409 updates unchanged.',
        scope='Timing-only development exploration; original loss definitions, training-only calibration, no per-pair hyperparameters.')
    for key in ('recipe','scope','evidence'):
        if key in decision:plan[key]=decision[key]
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        original_core_sha256=digest(reference),changed_core_sha256=digest(stage/CORE),
        changes='Multiply auxiliary scales by frozen envelope. Original forward, losses, backward and AdamW step preserved; extra scalar multiplications charged.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(source.MODULE,MODULE).replace('hippo-unetr-combos',decision.get('slurm_job_name','hippo-unetr-window')))
    pm=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/source.NAME/f.relative_to(stage)) for f in stage.rglob('*')
            if f.is_file() and pm.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/source.NAME/'manifest.json')
    for reference in references:
        origin=reference['origin'].split('/reports/',1)[1]
        for f in (stage/'persistent_references'/reference['name']).iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':
                donors[str(f.relative_to(stage))]=str(REMOTE/origin/f.name)
    for c in CANDIDATES:
        for f in (stage/'protocols'/c['key']).iterdir():
            if f.name not in ('protocol.json','parent_protocol.json'):
                donors[str(f.relative_to(stage))]=str(REMOTE/source.ORIGINAL/'protocols'/('unetr_vit_b16_'+c['arm'])/f.name)
    write(stage/'deployment_known_donors.json',donors);campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    manifest=read(stage/'manifest.json')
    for name,sha in manifest.items(): assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['num_runs']==36 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    parent=read(stage/'parent_manifest.json');original=read(stage/'original_manifest.json')
    assert read(stage/'optimizer_evidence.json')['outcome']=='diagnostic_replay_verified'
    reference=stage/'reference_source/capped_core.py'
    assert digest(reference)==parent[CORE] and (stage/CORE).read_text()==window.instrument_source(reference.read_text())
    for name,sha in parent.items():
        if name.startswith('source/') and name not in (CORE,'source/source_manifest.json'):
            assert manifest[name]==sha,name
    for c in CANDIDATES:
        folder=stage/'protocols'/c['key'];p=read(folder/'protocol.json')
        assert manifest[f"protocols/{c['key']}/parent_protocol.json"]==original[f"protocols/unetr_vit_b16_{c['arm']}/protocol.json"]
        assert p==protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),c)
        s=p['models']['unet']['objectives'][c['arm']]
        assert s['max_steps']==409 and s['prefix_updates']==102
        assert campaign.training_flops(409,s)<=CAP<campaign.training_flops(410,s)
        assert p['calibration_target_ratio']==c.get('target',.05) and p['calibration_p95_cap']==.75
        assert p['pkl_sha256']==manifest['data/MNI.pkl']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert manifest[f"protocols/{c['key']}/{name}"]==p[key]
    controls=read(stage/'controls.json');assert {(c['seed'],c['fold']) for c in controls}==set(PAIRS)
    for c in controls:
        folder=stage/'controls'/c['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==c['manifest_sha256']
        cfg=read(folder/'config.json')
        assert cfg['max_steps']==409 and cfg['budget_flops']==CAP and cfg['script_sha256']==digest(reference)
    for item in read(stage/'persistent_references.json'):
        folder=stage/'persistent_references'/item['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==item['manifest_sha256']
    return plan


def selection(stage,plan):
    baselines={(c['seed'],c['fold']):c['validation'] for c in read(stage/'controls.json')}
    rows=[]
    for task in plan['tasks']:
        folder=stage/'runs'/task['id']
        if not (folder/'cleanup.json').exists(): continue
        value=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        rows.append(dict(**task,validation=value,gain_pp=100*(value-baselines[(task['seed'],task['fold'])])))
    return select(plan['tasks'],rows,KEYS)


def eligible(stage,plan,state):
    return state['pending']


def preflight(stage):
    verify(stage)
    import ast
    import torch,monai
    from types import SimpleNamespace
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    ast.parse((stage/CORE).read_text())
    folder=stage/'protocols'/KEYS[0]
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective=CANDIDATES[0]['arm'],fold=0))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape)==[1,4,64,64,64]
    assert sum(p.numel() for p in model.parameters())==92685572
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');p=read(folder/'protocol.json')
    checks=[]
    for fold,split in enumerate(read(folder/'splits_10patients.json')):
        data=core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
        assert len(data)==20 and len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
        item=data[0];label=item['label'][None];torch.manual_seed(811)
        z=torch.randn((1,4,64,64,64),requires_grad=True)
        sdf=objectives.signed_distance_maps(label)
        depth=objectives.geometry_weights(label,p['native_geometry'][item['case_name']]['model_affine'],p['depth_geometry'])
        base=objectives.base_loss(z,label,'unetr_vit_b16');gb=torch.autograd.grad(base,z,retain_graph=True)[0]
        for arm in sorted({c['arm'] for c in CANDIDATES}):
            weights={name:.02 for name in objectives.TERMS if name in arm}
            zero=objectives.training_loss(z,label,arm,weights,0.,sdf,None,depth,0.,identity='unetr_vit_b16')
            assert torch.equal(zero,base) and torch.equal(torch.autograd.grad(zero,z,retain_graph=True)[0],gb)
            half=objectives.training_loss(z,label,arm,weights,.5,sdf,None,depth,.5,identity='unetr_vit_b16')
            assert torch.isfinite(half) and torch.isfinite(torch.autograd.grad(half,z,retain_graph=True)[0]).all()
            calibration=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
            targets=sorted({c.get('target',.05) for c in CANDIDATES if c['arm']==arm})
            for target in targets:
                selected,statistics=objectives.select_weights([calibration],arm,target,.75)
                assert all(v['weighted_median_ratio']<=target+1e-7 and v['weighted_p95_ratio']<=.7500001 for v in statistics.values())
                value=objectives.training_loss(z,label,arm,selected,1.,sdf,None,depth,1.,identity='unetr_vit_b16')
                assert torch.isfinite(value) and torch.isfinite(torch.autograd.grad(value,z,retain_graph=True)[0]).all()
            checks.append(dict(fold=fold,arm=arm,zero_weight_matches_base=True,calibration_targets_checked=targets))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,
        checks=checks,original_loss_definitions=True,scalar_operations_accounted=True,maximum_updates=409)
    write(stage/'preflight.json',result);return result


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=()
    runner.verify,runner.selection,runner.eligible=verify,selection,eligible
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train;batch.run=runner.run
    campaign.main()


if __name__=='__main__': main()
