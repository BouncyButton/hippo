"""Training-only target calibration: three-fold pilots, then nine-pair confirmation."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from statistics import mean
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,headroom_bands,seed_recovery as runner
from scripts.mni_constraints.shared43_study import ROOT,NAME as PARENT,CAP
from scripts.unetr_subfields import cap_extension as batch

NAME='mni_unetr_stronger_bands_shared43t_20261108'
MODULE='scripts.mni_constraints.unetr_strength'
TARGETS=(.10,.20,.40)


def key(target):return f'unetr_bands_t{round(target*100):02d}'


def tasks():
    return [dict(id=f'MNI_unetr_bands_t{round(target*100):02d}_f{fold}s{seed}',kind='conventional',
                 dataset='MNI',model='unetr_vit_b16',arm='dice_bands',fold=fold,seed=seed,
                 protocol=key(target),target=target,augmentation='xy_d4_v1')
            for seed in range(3) for target in TARGETS for fold in range(3)]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    for name,sha in read(parent/'manifest.json').items():assert digest(parent/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('unetr_strength.py','seed_recovery.py','collect_seed_recovery.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    origin=parent/'protocols/unetr_vit_b16_dice_bands'
    for target in TARGETS:
        folder=stage/'protocols'/key(target);shutil.copytree(origin,folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        p=read(folder/'protocol.json')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),calibration_target_ratio=target,
                 decision_rule='Targets10%,20%,40% each on seed0 folds0..2; choose largest positive mean final-budget gain over matched baseline, lower target breaks ties. Confirm selected target on seeds1,2 all folds, yielding nine matched pairs. Retain all pilot results.',
                 calibration_strength_policy='Fresh all20 training-crop calibration for each run; fixed75% p95 guard; no coefficient multiplication or validation optimization.')
        write(folder/'protocol.json',p)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    controls=[]
    for arm in ('dice','dice_bands'):
        for seed in range(3):
            for fold in range(3):
                identifier=f'MNI_unetr_vit_b16_{arm}_f{fold}s{seed}'
                src=ROOT/'reports'/PARENT/('frozen/reused_runs' if fold==seed==0 else 'runs')/identifier
                verify_archive(src,read(src/'archive_manifest.json'))
                assert read(src/'cleanup.json')['status']=='complete'
                shutil.copytree(src,stage/'controls'/identifier)
                controls.append(dict(id=identifier,arm=arm,fold=fold,seed=seed,
                    validation_dice=read(src/'validation_latest.json')['summary']['macro_hard_dice'],
                    training_dice=read(src/'training_final.json')['summary']['macro_hard_dice'],
                    archive_manifest_sha256=digest(src/'archive_manifest.json')))
    write(stage/'controls.json',controls)
    plan=dict(name=NAME,tasks=tasks(),num_runs=27,budget_flops=CAP,gpu_workers=2,
              augmentation='xy_d4_v1',models=['unetr_vit_b16'],folds=[0,1,2],seeds=[0,1,2],
              targets=list(TARGETS),prerequisite_stage=str(REMOTE/runner.NAME),
              conventional_python=read(parent/'plan.json')['conventional_python'],
              graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
              storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
              selection_rule=p['decision_rule'],
              selection_caveat='Exploratory development-validation target selection; overlapping folds are not independent. Consistent improvement requires reporting all nine matched gains; call consistent only if all nine improve.',
              authorization='After baseline seed recovery, investigate stronger UNETR bands targets with unchanged recipes, splits, augmentation, cap and75% p95 guard; confirm promising target across nine matched pairs. Finite two-worker job with per-run verified archival and cleanup.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          changes='Frozen training runtime unchanged; vary only training-calibration target. Adaptive three-fold pilot then nine-pair confirmation.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.shared43_study',MODULE).replace('hippo-mni-14','hippo-unetr-strength'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    parent=read(stage/'parent_manifest.json');plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['budget_flops']==CAP and plan['gpu_workers']==2
    allowed={'name','data','calibration_target_ratio','decision_rule','calibration_strength_policy'}
    for target in TARGETS:
        folder=stage/'protocols'/key(target);p=read(folder/'protocol.json');old=read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json')==parent['protocols/unetr_vit_b16_dice_bands/protocol.json']
        assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
        assert p['calibration_target_ratio']==target and p['calibration_p95_cap']==.75
        assert p['budget_flops']==CAP and p['folds']==p['seeds']==[0,1,2]
        for name,k in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[k]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for c in read(stage/'controls.json'):
        folder=stage/'controls'/c['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==c['archive_manifest_sha256']
    return plan


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    folder=stage/'protocols'/key(TARGETS[0]);p=read(folder/'protocol.json')
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective='dice_bands',fold=0))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    output=model(torch.zeros((1,1,64,64,64),device='meta'));profile=read(folder/'profile.json')
    assert [list(output.shape)]==profile['output_shapes']
    assert sum(x.numel() for x in model.parameters())==profile['parameters']
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl')
    data=core.PreparedSubfields(frame,read(folder/'splits_10patients.json')[0]['train'],DATASETS['MNI'])
    assert len(data)==20
    label=data[0]['label'][None];torch.manual_seed(0)
    raw=torch.randn((1,4,64,64,64),requires_grad=True)
    row=objectives.calibration_case(raw,label,'dice_bands',identity='unetr_vit_b16')
    checks=[]
    for target in TARGETS:
        weights,stats=objectives.select_weights([row],'dice_bands',target,.75)
        assert 0<stats['bands']['weighted_median_ratio']<=target+1e-8
        assert stats['bands']['weighted_p95_ratio']<=.75000001
        loss=objectives.training_loss(raw,label,'dice_bands',weights,1.,identity='unetr_vit_b16')
        assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,raw)[0]).all()
        checks.append(dict(target=target,synthetic_only_weights=weights,statistics=stats))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,checks=checks)
    write(stage/'preflight.json',result);return result


def selection(stage,plan):
    controls=read(stage/'controls.json')
    baseline={(r['fold'],r['seed']):r for r in controls if r['arm']=='dice'}
    old={(r['fold'],r['seed']):r for r in controls if r['arm']=='dice_bands'}
    rows=[]
    for task in plan['tasks']:
        folder=stage/'runs'/task['id']
        if not (folder/'cleanup.json').exists():continue
        val=read(folder/'validation_latest.json')['summary']['macro_hard_dice']
        train=read(folder/'training_final.json')['summary']['macro_hard_dice']
        pair=(task['fold'],task['seed'])
        rows.append(dict(**task,validation_dice=val,training_dice=train,
                    gain_vs_baseline_pp=100*(val-baseline[pair]['validation_dice']),
                    gain_vs_original5pct_pp=100*(val-old[pair]['validation_dice']),
                    calibration=read(folder/'calibration.json')['statistics']))
    pilots=[r for r in rows if r['seed']==0]
    chosen=None;summary=[]
    if len(pilots)==9:
        gains={t:mean(r['gain_vs_baseline_pp'] for r in pilots if r['target']==t) for t in TARGETS}
        promising=[t for t in TARGETS if gains[t]>0]
        if promising:chosen=max(promising,key=lambda t:(gains[t],-t))
    for target in TARGETS:
        group=[r for r in rows if r['target']==target]
        if group:
            summary.append(dict(target=target,cells=len(group),mean_gain_pp=mean(r['gain_vs_baseline_pp'] for r in group),
                        improved_cells=sum(r['gain_vs_baseline_pp']>0 for r in group),
                        consistent_nine_pair_improvement=len(group)==9 and all(r['gain_vs_baseline_pp']>0 for r in group)))
    complete=len(pilots)==9 and (chosen is None or sum(r['target']==chosen for r in rows)==9)
    return dict(rows=rows,summary=summary,selected=dict(target=chosen),complete=complete,
                rule=plan['selection_rule'],caveat=plan['selection_caveat'])


def eligible(stage,plan,state):
    completed={r['id'] for r in state['rows']}
    pilots=[t for t in plan['tasks'] if t['seed']==0 and t['id'] not in completed]
    if pilots:return pilots
    chosen=state['selected']['target']
    return [t for t in plan['tasks'] if t['target']==chosen and t['id'] not in completed]


def run(stage):
    plan=verify(stage)
    assert read(Path(plan['prerequisite_stage'])/'selection.json')['complete'],'Recover working baseline seeds first'
    assert read(Path(plan['prerequisite_stage'])/'status.json')['status']=='complete'
    runner.run(stage)


def main():
    runner.MODULE=MODULE;runner.RETIREMENTS=()
    runner.verify=verify;runner.selection=selection;runner.eligible=eligible
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=headroom_bands.train
    batch.run=run
    campaign.main()


if __name__=='__main__':main()
