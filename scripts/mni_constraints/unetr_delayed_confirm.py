"""Nine matched UNETR pairs with one frozen delayed5% bands recipe."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign, headroom_bands
from scripts.mni_constraints.unetr_constraint_pilot import NAME as PILOT, PARENT, CAP, ROOT
from scripts.unetr_subfields import cap_extension as batch

NAME = 'mni_unetr_delayed05_confirm53t_20261108'
MODULE = 'scripts.mni_constraints.unetr_delayed_confirm'
SEEDS = (0, 5, 19)
ARMS = {'baseline': 'dice', 'delayed_t05': 'dice_bands'}


def tasks():
    return [dict(id=f'MNI_unetr_{key}_f{fold}s{seed}', kind='conventional', dataset='MNI',
                 model='unetr_vit_b16', arm=arm, protocol=key, fold=fold, seed=seed,
                 augmentation='xy_d4_v1')
            for seed in SEEDS for fold in range(3) if (fold,seed) != (0,19)
            for key,arm in ARMS.items()]


def changed_protocol(old):
    p = deepcopy(old)
    p.update(name=NAME, data=str(REMOTE/NAME/'data/MNI.pkl'), folds=[0,1,2], seeds=list(SEEDS),
        primary_metric='Final-budget validation macro Dice, paired delayed5% minus baseline in each seed/fold.',
        decision_rule='Complete all eight new matched pairs plus the verified seed19/fold0 pilot pair. Retain all scores; no seed replacement, early stopping or retuning. Report all nine and the eight new pairs separately.',
        selection_caveat='Recipe selected on seed19/fold0 development validation; that pair is reused, not independent confirmation. Folds overlap; no independent-test or significance claim.')
    return p


def prepare(_target=None):
    stage, parent = ROOT/'experiments'/NAME, ROOT/'experiments'/PILOT
    broad = ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():
        return verify(stage)
    for origin in (parent,broad):
        for name,sha in read(origin/'manifest.json').items():
            assert digest(origin/name) == sha, name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(Path(__file__),stage/'source/scripts/mni_constraints/unetr_delayed_confirm.py')
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    shutil.copy2(broad/'manifest.json',stage/'baseline_parent_manifest.json')
    origins = {'baseline':broad/'protocols/unetr_vit_b16_dice',
               'delayed_t05':parent/'protocols/delayed_t05'}
    for key,origin in origins.items():
        folder=stage/'protocols'/key
        shutil.copytree(origin,folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        write(folder/'protocol.json',changed_protocol(read(folder/'parent_protocol.json')))
    controls = {'baseline': ROOT/'reports'/PARENT/'runs/MNI_unetr_vit_b16_dice_f0s19',
                'delayed_t05': ROOT/'reports'/PILOT/'runs/MNI_unetr_delayed_t05_f0s19'}
    reuse=[]
    for key,origin in controls.items():
        verify_archive(origin,read(origin/'archive_manifest.json'))
        assert read(origin/'cleanup.json')['status']=='complete'
        shutil.copytree(origin,stage/'controls'/key)
        reuse.append(dict(condition=key,seed=19,fold=0,archive=str(origin),
                          manifest_sha256=digest(origin/'archive_manifest.json')))
    write(stage/'control_reuse.json',reuse)
    plan=dict(name=NAME,tasks=tasks(),num_runs=16,logical_runs=18,reused_runs=2,
        models=['unetr_vit_b16'],seeds=list(SEEDS),folds=[0,1,2],budget_flops=CAP,
        gpu_workers=2,excluded_nodes=['gnode04'],augmentation='xy_d4_v1',
        conventional_python=read(parent/'plan.json')['conventional_python'],
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User explicitly requests confirming delayed5% UNETR on seeds0,5,19 and all folds at the same cap using2 GPUs; keep the remaining sweep paused.',
        recipe='Original union bands; fresh all20-training-crop calibration after102 updates,target0.05,p95cap0.75,33-update ramp,fixed coefficient thereafter.409 updates; unchanged AdamW,LR0.0006,24 warmup,hold-half,dropout0.1,xyD4.',
        evaluation='Final-budget per-volume macro Dice primary; per-subfield and union Dice,FN,FP,internal swaps,train curves secondary. All9 paired gains and8 new confirmation gains reported separately.',
        decision_rule='No recipe/seed changes. Describe positive mean and number of positive pairs separately; consistent improvement requires all9 positive. The rest of the campaign remains paused regardless of result.',
        selection_caveat='Pilot pair reused. Overlapping folds and development validation are not independent test samples.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        changes='Training runtime byte-identical to completed pilot; expand only seeds/folds and pair against the same53T baseline.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.unetr_constraint_pilot',MODULE).replace('hippo-unetr-probe','hippo-unetr-confirm'))
    original=read(parent/'manifest.json')
    donors={str(f.relative_to(stage)):str(REMOTE/PILOT/f.relative_to(stage))
            for f in stage.rglob('*') if f.is_file() and original.get(str(f.relative_to(stage)))==digest(f)}
    donors.pop('source/source_manifest.json',None)
    donors['parent_manifest.json']=str(REMOTE/PILOT/'manifest.json')
    for key,origin in origins.items():
        remote_origin=(REMOTE/PARENT/'protocols/unetr_vit_b16_dice' if key=='baseline'
                       else REMOTE/PILOT/'protocols/delayed_t05')
        for f in (stage/'protocols'/key).iterdir():
            if f.name not in ('protocol.json','parent_protocol.json'):
                donors[str(f.relative_to(stage))]=str(remote_origin/f.name)
    write(stage/'deployment_known_donors.json',donors)
    campaign.freeze(stage)
    return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json')
    assert plan['tasks']==tasks() and plan['num_runs']==16 and plan['gpu_workers']==2
    assert plan['budget_flops']==CAP and plan['seeds']==list(SEEDS) and plan['folds']==[0,1,2]
    original=read(stage/'parent_manifest.json');broad=read(stage/'baseline_parent_manifest.json')
    for name,sha in original.items():
        if name.startswith('source/') and name!='source/source_manifest.json':
            assert digest(stage/name)==sha,name
    protocols={}
    for key,arm in ARMS.items():
        folder=stage/'protocols'/key
        expected=(broad['protocols/unetr_vit_b16_dice/protocol.json'] if key=='baseline'
                  else original['protocols/delayed_t05/protocol.json'])
        assert digest(folder/'parent_protocol.json')==expected
        p=read(folder/'protocol.json');assert p==changed_protocol(read(folder/'parent_protocol.json'))
        protocols[key]=p
        s=p['models']['unet']['objectives'][arm]
        assert p['budget_flops']==CAP and p['lr_horizon_updates']==s['max_steps']==409
        assert campaign.training_flops(409,s)<=CAP<campaign.training_flops(410,s)
        for name,field in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),
                           ('nnunet_architecture.json','nnunet_architecture_sha256')]:
            assert digest(folder/name)==p[field]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
        if key=='delayed_t05':
            assert s['prefix_updates']==102 and s['constraint_warmup_updates']==33
            assert p['calibration_target_ratio']==.05 and p['calibration_p95_cap']==.75
        control=stage/'controls'/key
        verify_archive(control,read(control/'archive_manifest.json'))
        cfg=read(control/'config.json')
        assert cfg['seed']==19 and cfg['fold']==0 and cfg['schedule']==s
        assert cfg['script_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
    for field in ('initial_lrs','lr_horizon_updates','lr_schedule_details','augmentation','dropout_rate',
                  'loss_arguments','adamw_weight_decay','pkl_sha256','splits_sha256','native_geometry'):
        assert protocols['baseline'][field]==protocols['delayed_t05'][field],field
    return plan


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS,subject
    from scripts.mni_constraints import objectives
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    folder=stage/'protocols/delayed_t05'
    batch.bind(SimpleNamespace(protocol_dir=folder,model='unet',objective='dice_bands',fold=0))
    model=core.base.build_model('unet',read(folder/'nnunet_architecture.json')).to('meta').train()
    assert list(model(torch.zeros((1,1,64,64,64),device='meta')).shape)==[1,4,64,64,64]
    assert sum(p.numel() for p in model.parameters())==92685572
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');checks=[]
    for fold,split in enumerate(read(folder/'splits_10patients.json')):
        train=core.PreparedSubfields(frame,split['train'],DATASETS['MNI'])
        val=core.PreparedSubfields(frame,split['val'],DATASETS['MNI'])
        assert len(train)==20 and len(val)==10
        assert len({subject(c) for c in split['train']})==10
        assert not {subject(c) for c in split['train']}&{subject(c) for c in split['val']}
        torch.manual_seed(0);z=torch.randn((1,4,64,64,64),requires_grad=True);label=train[0]['label'][None]
        row=objectives.calibration_case(z,label,'dice_bands',identity='unetr_vit_b16')
        weights,stats=objectives.select_weights([row],'dice_bands',.05,.75)
        loss=objectives.training_loss(z,label,'dice_bands',weights,1.,identity='unetr_vit_b16')
        assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,z)[0]).all()
        checks.append(dict(fold=fold,training_crops=20,validation_crops=10,synthetic_statistics=stats))
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),optimizer_updates=0,
                checks=checks,unchanged_training_runtime=True,new_runs=16,reused_runs=2)
    write(stage/'preflight.json',result);return result


def main():
    campaign.MODULE=MODULE
    campaign.prepare,campaign.verify,campaign.preflight=prepare,verify,preflight
    campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
