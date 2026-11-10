"""Fixed two-pair timing/strength factorial; original bands and base unchanged."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.prepare import schedule
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.unetr_subfields import cap_extension as batch

ROOT=campaign.ROOT
NAME='mni_unet_bands_timing_strength53t_f1s05_20261109'
PARENT='mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
MODULE='scripts.mni_constraints.unet_bands_timing_strength'
CAP=53305109592450
VARIANTS={'early10':(False,.10),'early05':(False,.05),
          'late10':(True,.10),'late05':(True,.05)}


def tasks():
    return [dict(id=f'MNI_unet_{variant}_f1s{seed}',kind='conventional',dataset='MNI',
                 model='basic_unet16',arm='dice_bands',fold=1,seed=seed,protocol=variant,
                 target=VARIANTS[variant][1],augmentation='none')
            for variant in ('late05','late10','early05') for seed in (5,0)]


def changed_protocol(old,profile,variant):
    p=deepcopy(old);costs=deepcopy(profile['costs'])
    for arm,cost in costs.items():cost['update_flops']+=int('bands' in arm)+int('edge' in arm)
    delay,target=VARIANTS[variant]
    half=CAP/(2*costs['dice']['update_flops'])
    late=min((math.floor(half),math.ceil(half)),key=lambda n:(abs(n*costs['dice']['update_flops']-CAP/2),n))
    prefix=late if delay else 76
    s=schedule(costs,'dice_bands',dict(prefix_updates=prefix,sdf_prefix_updates=prefix,
               constraint_warmup_updates=76),20,CAP)
    p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),folds=[1],seeds=[0,5],
        models={'unet':dict(parameters=profile['parameters'],objectives={'dice_bands':s})},
        lr_horizon_updates=s['max_steps'],calibration_target_ratio=target,
        lr_schedule_details=dict(policy='cap_matched',horizon_updates=s['max_steps'],
                                 warmup_updates=math.ceil(.05*s['max_steps']),decay_start_index=math.ceil(.05*s['max_steps'])),
        decision_rule='Run every predeclared pilot cell. Compare final-budget Dice and CA4/DG with matched baseline and existing early10. No early rejection, seed replacement, or automatic recipe promotion.',
        timing_policy=f'Training-only calibration after {prefix} successful updates; original bands ramps over76 updates. Delay chosen from compute only, not validation.',
        calibration_strength_policy=f'Fresh all20 training-crop calibration at activation; target{target},p95cap0.75. Fixed coefficient thereafter.',
        selection_caveat='Two development pairs only: seed5/fold1 failure and seed0/fold1 positive control. A fix requires positive paired gains on both, then separate wider confirmation.')
    return p


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    original=read(parent/'manifest.json')
    for name,sha in original.items():assert digest(parent/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    dest=stage/'source/scripts/mni_constraints/unet_bands_timing_strength.py'
    assert not dest.exists();shutil.copy2(Path(__file__),dest)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    origin=parent/'protocols/basic_unet16_dice_bands'
    for variant in VARIANTS:
        folder=stage/'protocols'/variant;shutil.copytree(origin,folder)
        shutil.copy2(origin/'protocol.json',folder/'parent_protocol.json')
        write(folder/'protocol.json',changed_protocol(read(origin/'protocol.json'),read(origin/'profile.json'),variant))
    controls=[];donors={}
    for seed in (0,5):
        for arm in ('dice','dice_bands'):
            identity=f'MNI_basic_unet16_{arm}_f1s{seed}'
            source=ROOT/'reports'/PARENT/'runs'/identity
            verify_archive(source,read(source/'archive_manifest.json'))
            assert read(source/'cleanup.json')['status']=='complete'
            target=stage/'controls'/identity;shutil.copytree(source,target)
            controls.append(dict(id=identity,seed=seed,fold=1,arm=arm,
                archive_sha256=digest(source/'archive_manifest.json'),source_archive=str(source)))
            for f in target.iterdir():
                if f.is_file() and f.name!='archive_acknowledgment.json':
                    donors[str(f.relative_to(stage))]=str(REMOTE/PARENT/'runs'/identity/f.name)
    write(stage/'controls.json',controls)
    plan=dict(name=NAME,tasks=tasks(),num_runs=6,logical_cells=10,reused_runs=4,
        models=['basic_unet16'],folds=[1],seeds=[0,5],budget_flops=CAP,gpu_workers=2,
        augmentation='none',conditions=list(VARIANTS),conventional_python=read(parent/'plan.json')['conventional_python'],
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=2,archive=str(ROOT/'reports'/NAME)),
        authorization='User authorized delayed activation and lower strength on seed5/fold1 and seed0/fold1 at the established FLOPs cap without changing original bands. Six fresh factorial cells, four exact archived controls, two workers. Temporarily interrupt and then resume the full matrix with its recipes unchanged.',
        recipe_policy='No automatic promotion or full-matrix modification; report all pilot outcomes.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        unchanged_objective_sha256=digest(stage/'source/scripts/mni_constraints/objectives.py'),
        unchanged_core_sha256=digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py'),
        changes='Only protocol timing and calibration strength change. Frozen original objectives, training core, base recipe, data, sampling and diagnostics are unchanged. Existing exact graph FLOP profiles are reused; piecewise schedules recomputed.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.verified_lower_lr_full_matrix',MODULE).replace('hippo-verified53','hippo-unet-delay'))
    for f in stage.rglob('*'):
        if f.is_file() and original.get(str(f.relative_to(stage)))==digest(f):
            donors[str(f.relative_to(stage))]=str(REMOTE/PARENT/f.relative_to(stage))
    donors.pop('source/source_manifest.json',None)
    donors['parent_manifest.json']=str(REMOTE/PARENT/'manifest.json')
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    original=read(stage/'parent_manifest.json');plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['budget_flops']==CAP and plan['gpu_workers']==2
    for name,sha in original.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for variant in VARIANTS:
        folder=stage/'protocols'/variant;p=read(folder/'protocol.json');old=read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json')==original['protocols/basic_unet16_dice_bands/protocol.json']
        assert p==changed_protocol(old,read(folder/'profile.json'),variant)
        assert p['initial_lrs']=={'unet':.00016} and p['constraints']==['bands'] and p['augmentation']=='none'
        for file,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/file)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        s=p['models']['unet']['objectives']['dice_bands'];n=s['max_steps']
        assert campaign.training_flops(n,s)<=CAP<campaign.training_flops(n+1,s)
        assert s['prefix_updates']+76<n and s['calibration_flops']==382099497372
        if variant=='early10':assert s==old['models']['unet']['objectives']['dice_bands']
    for control in read(stage/'controls.json'):
        folder=stage/'controls'/control['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
        assert digest(folder/'archive_manifest.json')==control['archive_sha256']
        cfg=read(folder/'config.json');assert cfg['budget_flops']==CAP and cfg['fold']==1 and cfg['seed']==control['seed']
        assert cfg['initial_lr']==.00016 and cfg['augmentation']=='none'
    return plan


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.mni_constraints import objectives
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    folder=stage/'protocols/early05';p=read(folder/'protocol.json');profile=read(folder/'profile.json')
    assert digest(Path(objectives.__file__))==profile['objective_sha256']
    trainer=batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective='dice_bands',fold=1))
    with torch.device('meta'):model=trainer.base.build_model('unet',read(folder/'nnunet_architecture.json'))
    outputs=model(torch.zeros((1,1,64,64,64),device='meta'))
    assert [list(outputs.shape)]==profile['output_shapes'] and sum(x.numel() for x in model.parameters())==profile['parameters']
    criterion,optimizer=trainer.base.build_loss_and_optimizer('unet',model,p)
    assert isinstance(optimizer,torch.optim.AdamW) and optimizer.param_groups[0]['lr']==.00016
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');split=read(folder/'splits_10patients.json')[1]
    train=core.PreparedSubfields(frame,split['train'],DATASETS['MNI']);assert len(train)==20
    assert len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
    torch.manual_seed(17);z=torch.randn((1,4,64,64,64),requires_grad=True);label=train[0]['label'][None]
    row=objectives.calibration_case(z,label,'dice_bands',identity='basic_unet16');checks=[]
    for variant,(delay,target) in VARIANTS.items():
        weights,stats=objectives.select_weights([deepcopy(row)],'dice_bands',target,.75)
        loss=objectives.training_loss(z,label,'dice_bands',weights,1.,identity='basic_unet16')
        assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,z)[0]).all()
        assert abs(stats['bands']['weighted_median_ratio']-target)<1e-8
        s=read(stage/'protocols'/variant/'protocol.json')['models']['unet']['objectives']['dice_bands']
        checks.append(dict(variant=variant,target=target,schedule=s,synthetic_weight=weights['bands']))
    assert math.isclose(checks[1]['synthetic_weight']*2,checks[0]['synthetic_weight'])
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=checks,
                optimizer_updates=0,parameters=profile['parameters'],torch=str(torch.__version__),monai=monai.__version__)
    write(stage/'preflight.json',result);return result


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify;campaign.preflight=preflight
    campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
