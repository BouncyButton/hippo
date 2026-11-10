"""Factorial timing study: original constraints after training-only readiness."""
import csv
from copy import deepcopy
from pathlib import Path
import shutil
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands, seed_recovery as runner
from scripts.mni_constraints import unetr_delayed_combinations as previous
from scripts.mni_constraints.unetr_readiness import onset, TRAINING_FIELDS
from scripts.mni_constraints.final53_study import changed_protocol
from scripts.unetr_subfields import cap_extension as batch

ROOT, CAP, SEEDS, PAIRS, PILOTS = previous.ROOT, previous.CAP, previous.SEEDS, previous.PAIRS, previous.PILOTS
PARENT = previous.NAME
NAME = 'mni_unetr_readiness53t_20261108'
MODULE = 'scripts.mni_constraints.unetr_readiness_study'
CANDIDATES = tuple(dict(key=f'ready{threshold}_{label}', threshold=threshold/100, arm=arm)
    for threshold in (50, 60)
    for label, arm in (('bands', 'dice_bands'), ('full', 'dice_bands_edge_sdf_depth')))
KEYS = tuple(c['key'] for c in CANDIDATES)


def tasks():
    pairs = (*PILOTS, *(p for p in PAIRS if p not in PILOTS))
    return [dict(id=f"MNI_unetr_{c['key']}_f{f}s{s}", candidate=c['key'],
                 threshold=c['threshold'], arm=c['arm'], protocol=f"{c['key']}_f{f}s{s}",
                 kind='conventional', dataset='MNI', model='unetr_vit_b16',
                 seed=s, fold=f, augmentation='xy_d4_v1', pilot=(s,f) in PILOTS)
            for s,f in pairs for c in CANDIDATES]


def derive(folder, threshold):
    cfg = read(folder/'config.json')
    with (folder/'metrics.csv').open() as stream:
        curve = list(csv.DictReader(stream))
    return dict(update=onset(curve, threshold, CAP, cfg['schedule']['baseline_flops_per_update'],409),
                threshold=threshold, consecutive_regular_checks=2, fields=list(TRAINING_FIELDS),
                metrics_sha256=digest(folder/'metrics.csv'))


def protocol(old, profile, task, evidence):
    p = changed_protocol(old, profile)
    arm = task['arm']; prefix = evidence['update']; costs = deepcopy(profile['costs'])
    for a,c in costs.items():
        c['update_flops'] += int('bands' in a) + int('edge' in a)
    timing = dict(prefix_updates=prefix, constraint_warmup_updates=33,
                  sdf_prefix_updates=prefix+int('sdf' in arm))
    sched = schedule(costs, arm, timing, 20, CAP)
    assert sched['max_steps'] == 409 and prefix+33 <= 409
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'),
             seeds=[task['seed']], folds=[task['fold']],
             models={'unet':dict(parameters=profile['parameters'], objectives={arm:sched})},
             constraint_warmup_updates=33, readiness=evidence,
             timing_policy='Original constraints activate after all training subfields meet one shared threshold at two consecutive regular5%-budget checks. Fresh training-only calibration at that point;33-update ramp. SDF activates no earlier than both readiness and nearest5/9 cap.',
             decision_rule='One training-only rule per candidate, identical across seeds and folds. Candidate selection uses four fixed development pairs, then five additional pairs; every gain must be positive.',
             selection_caveat='Adaptive development-validation exploration on overlapping folds; no independent test-generalization claim. Previous fixed-onset recipes failed consistency.',
             primary_metric='Final409-update macro Dice gain over matched unconstrained baseline.',
             readiness_caveat='Onset is precomputed exclusively from matched baseline TRAINING columns; verify identical constrained prefix. No validation-based onset choice.')
    return p


def prepare(_target=None):
    report = ROOT/'reports'/PARENT
    outcome = read(report/'combination_results.json')
    assert outcome['status']=='complete' and outcome['outcome']=='exhausted', 'Finish fixed combination pilots first'
    stage = ROOT/'experiments'/NAME; parent = ROOT/'experiments'/PARENT
    original = ROOT/'experiments'/previous.ORIGINAL
    if (stage/'manifest.json').exists():
        return verify(stage)
    previous.verify(parent)
    for part in ('source','data','controls'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for filename in ('unetr_readiness_study.py','unetr_readiness.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/filename,stage/'source/scripts/mni_constraints'/filename)
    for filename in ('controls.json','original_manifest.json'):
        shutil.copy2(parent/filename,stage/filename)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    write(stage/'previous_results.json',outcome)
    controls={(c['seed'],c['fold']):stage/'controls'/c['name'] for c in read(stage/'controls.json')}
    records=[]
    for task in tasks():
        folder=stage/'protocols'/task['protocol']
        shutil.copytree(original/'protocols'/('unetr_vit_b16_'+task['arm']),folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        evidence=derive(controls[(task['seed'],task['fold'])],task['threshold'])
        write(folder/'protocol.json',protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),task,evidence))
        records.append(dict(task=task['id'],candidate=task['candidate'],seed=task['seed'],fold=task['fold'],**evidence))
    write(stage/'onsets.json',records)
    plan=deepcopy(read(parent/'plan.json'))
    plan.update(name=NAME,tasks=tasks(),num_runs=36,maximum_new_runs=36,pilot_runs=16,
        candidates=list(CANDIDATES),conditions=list(KEYS),archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='Continue the explicitly authorized goal after delayed fixed-onset combinations failed. Test two shared training-only readiness thresholds crossed with bands versus the original full combination, two GPUs, unchanged53.305T cap/base recipe/seeds/folds.',
        evidence='Fixed102-update onset precedes stable training classification. All original delayed combinations failed the consistency criterion on seed0/fold1. This tests timing without changing any constraint definition.',
        selection_rule='Complete16 fixed pilots. Require4/4 positive gains, rank worst-pair gain then mean then simpler candidate. Confirm each eligible candidate on all5 other pairs; success requires9/9, otherwise global research continues.',
        recipe='Original bands or original bands+edge+SDF+depth, target.05,p95.75; readiness50% or60% for every training subfield on two regular checks; fresh calibration,33-update ramp; SDF no earlier than nearest5/9 cap;409updates.',
        scope='Factorial development-validation exploration; no seed-specific tuning or replacement; onset reads no validation columns.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Training/loss code unchanged; add training-only readiness derivation and adaptive study scheduler.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(previous.MODULE,MODULE).replace('hippo-unetr-combos','hippo-unetr-ready'))
    parent_manifest=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/PARENT/f.relative_to(stage)) for f in stage.rglob('*')
            if f.is_file() and parent_manifest.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None);donors['parent_manifest.json']=str(REMOTE/PARENT/'manifest.json')
    for task in tasks():
        for f in (stage/'protocols'/task['protocol']).iterdir():
            if f.name not in ('protocol.json','parent_protocol.json'):
                donors[str(f.relative_to(stage))]=str(REMOTE/previous.ORIGINAL/'protocols'/('unetr_vit_b16_'+task['arm'])/f.name)
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    manifest=read(stage/'manifest.json')
    for name,sha in manifest.items():
        assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['num_runs']==36 and plan['gpu_workers']==2 and plan['budget_flops']==CAP
    assert read(stage/'previous_results.json')['outcome']=='exhausted'
    for name,sha in read(stage/'parent_manifest.json').items():
        if name.startswith('source/') and name!='source/source_manifest.json':
            assert manifest[name]==sha,name
    original=read(stage/'original_manifest.json');controls={}
    for c in read(stage/'controls.json'):
        folder=stage/'controls'/c['name'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==c['manifest_sha256']
        cfg=read(folder/'config.json')
        assert cfg['seed']==c['seed'] and cfg['fold']==c['fold'] and cfg['max_steps']==409 and cfg['budget_flops']==CAP
        assert cfg['script_sha256']==manifest['source/scripts/cross_dataset_campaign/capped_core.py']
        controls[(c['seed'],c['fold'])]=folder
    assert set(controls)==set(PAIRS)
    records=[]
    for task in tasks():
        folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        assert manifest[f"protocols/{task['protocol']}/parent_protocol.json"]==original[f"protocols/unetr_vit_b16_{task['arm']}/protocol.json"]
        evidence=derive(controls[(task['seed'],task['fold'])],task['threshold'])
        assert p==protocol(read(folder/'parent_protocol.json'),read(folder/'profile.json'),task,evidence)
        s=p['models']['unet']['objectives'][task['arm']]
        assert campaign.training_flops(409,s)<=CAP<campaign.training_flops(410,s)
        assert s['prefix_updates']==evidence['update'] and s['sdf_prefix_updates']>=s['prefix_updates']
        assert p['calibration_target_ratio']==.05 and p['calibration_p95_cap']==.75
        assert p['pkl_sha256']==manifest['data/MNI.pkl']
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert manifest[f"protocols/{task['protocol']}/{name}"]==p[key]
        records.append(dict(task=task['id'],candidate=task['candidate'],seed=task['seed'],fold=task['fold'],**evidence))
    assert read(stage/'onsets.json')==records
    return plan


def selection(stage,plan):
    baseline={(c['seed'],c['fold']):c['validation'] for c in read(stage/'controls.json')};rows=[]
    for t in plan['tasks']:
        folder=stage/'runs'/t['id']
        if not (folder/'cleanup.json').exists():continue
        value=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        rows.append(dict(**t,validation=value,gain_pp=100*(value-baseline[(t['seed'],t['fold'])])))
    summaries=[];ranking=[]
    for c in CANDIDATES:
        group=[r for r in rows if r['candidate']==c['key']];pilot=[r for r in group if r['pilot']]
        summaries.append(dict(candidate=c['key'],pairs=len(group),pilot_pairs=len(pilot),
            minimum_gain_pp=min((r['gain_pp'] for r in group),default=None),
            mean_gain_pp=mean(r['gain_pp'] for r in group) if group else None,
            all_nine_improved=len(group)==9 and all(r['gain_pp']>0 for r in group)))
        if len(pilot)==4 and all(r['gain_pp']>0 for r in pilot):
            ranking.append((min(r['gain_pp'] for r in pilot),mean(r['gain_pp'] for r in pilot),-len(c['arm'].split('_')),-c['threshold'],c['key']))
    ordered=[r[-1] for r in sorted(ranking,reverse=True)]
    winner=next((s['candidate'] for s in summaries if s['all_nine_improved']),None)
    pending=next((c for c in ordered if sum(r['candidate']==c for r in rows)<9),None)
    finished=sum(r['pilot'] for r in rows)==16 and (winner is not None or pending is None)
    return dict(rows=rows,summary=summaries,pilot_ranking=ordered,selected=dict(candidate=winner or pending,winner=winner),
                complete=finished,outcome='nine_pair_success' if winner else 'exhausted' if finished else 'running')


def eligible(stage,plan,state):
    completed={r['id'] for r in state['rows']}
    pilots=[t for t in plan['tasks'] if t['pilot'] and t['id'] not in completed]
    if pilots:return pilots
    if state['complete']:return []
    return [t for t in plan['tasks'] if t['candidate']==state['selected']['candidate'] and t['id'] not in completed]


def preflight(stage):
    verify(stage)
    import torch,monai
    from types import SimpleNamespace
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    task=tasks()[0];folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective=task['arm'],fold=task['fold']))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape)==[1,4,64,64,64]
    assert sum(x.numel() for x in model.parameters())==92685572
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');checks=[]
    for fold,split in enumerate(read(folder/'splits_10patients.json')):
        data=core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
        assert len(data)==20 and len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
        item=data[0];label=item['label'][None]
        depth=objectives.geometry_weights(label,p['native_geometry'][item['case_name']]['model_affine'],p['depth_geometry'])
        sdf=objectives.signed_distance_maps(label)
        for arm in ('dice_bands','dice_bands_edge_sdf_depth'):
            torch.manual_seed(731);z=torch.randn((1,4,64,64,64),requires_grad=True)
            row=objectives.calibration_case(z,label,arm,sdf,None,depth,identity='unetr_vit_b16')
            weights,stats=objectives.select_weights([row],arm,.05,.75)
            assert all(s['weighted_median_ratio']<=.0500001 and s['weighted_p95_ratio']<=.7500001 for s in stats.values())
            value=objectives.training_loss(z,label,arm,weights,1.,sdf,None,depth,1.,identity='unetr_vit_b16')
            assert torch.isfinite(value) and torch.isfinite(torch.autograd.grad(value,z)[0]).all()
            checks.append(dict(fold=fold,arm=arm))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,
                checks=checks,unchanged_training_runtime=True,baselines=9,
                readiness_source_sha256=digest(stage/'source/scripts/mni_constraints/unetr_readiness.py'),
                protocol_count=36,original_definitions=True)
    write(stage/'preflight.json',result)
    return result


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=()
    runner.verify,runner.selection,runner.eligible=verify,selection,eligible
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train;batch.run=runner.run
    campaign.main()


if __name__=='__main__':main()
