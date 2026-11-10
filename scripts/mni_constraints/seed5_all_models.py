"""Check seed5 as the common seed1 replacement under the frozen43T recipes."""
from pathlib import Path
import shutil
from statistics import mean
from types import SimpleNamespace

from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,headroom_bands
from scripts.mni_constraints.shared43_study import ROOT,NAME as PARENT,CAP,MODELS
from scripts.unetr_subfields import cap_extension as batch

NAME='mni_seed5_all_models_shared43t_20261108'
MODULE='scripts.mni_constraints.seed5_all_models'
RECOVERY='mni_seed_recovery_shared43t_20261108'


def tasks():
    return [dict(id=f'MNI_{m}_dice_f{f}s5',kind='conventional',dataset='MNI',model=m,
                 arm='dice',fold=f,seed=5,protocol=m+'_dice',augmentation='xy_d4_v1')
            for f in range(3) for m in ('swinunetr','unetr_vit_b16')]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    for name,sha in read(parent/'manifest.json').items():assert digest(parent/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('seed5_all_models.py','collect_seed5_all_models.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    reuse=[];controls=[]
    for m in MODELS:
        folder=stage/'protocols'/(m+'_dice');origin=parent/'protocols'/(m+'_dice')
        shutil.copytree(origin,folder);shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json')
        p=read(folder/'protocol.json')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),seeds=[5],
                 decision_rule='Test seed5 on all original folds0,1,2 for every model as a proposed common replacement for seed1. Mean final-budget Dice>=50%; separately report every-fold>=50%. Preserve seed1 and all failed attempts.',
                 primary_metric='Final-budget validation foreground macro Dice; equal mean over all three fixed folds.')
        write(folder/'protocol.json',p)
        for f in range(3):
            identifier=f'MNI_{m}_dice_f{f}s1'
            src=ROOT/'reports'/PARENT/'runs'/identifier
            verify_archive(src,read(src/'archive_manifest.json'))
            assert read(src/'cleanup.json')['status']=='complete'
            shutil.copytree(src,stage/'seed1_controls'/identifier)
            controls.append(dict(id=identifier,model=m,fold=f,seed=1,
                 archive_manifest_sha256=digest(src/'archive_manifest.json')))
            if m not in ('nnunet','basic_unet16'):continue
            identifier=f'MNI_{m}_dice_f{f}s5';src=ROOT/'reports'/RECOVERY/'runs'/identifier
            verify_archive(src,read(src/'archive_manifest.json'))
            assert read(src/'cleanup.json')['status']==read(src/'completion.json')['status']=='complete'
            cfg=read(src/'config.json')
            assert cfg['schedule']==p['models'][p['trainer_alias']]['objectives']['dice']
            assert cfg['initial_lr']==p['initial_lrs'][p['trainer_alias']]
            assert cfg['lr_horizon_updates']==p['lr_horizon_updates']
            assert cfg['augmentation']==p['augmentation'] and cfg['budget_flops']==CAP
            assert cfg['dataset_sha256']==digest(stage/'data/MNI.pkl')
            split=read(folder/'splits_10patients.json')[f]
            assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
            assert cfg['script_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
            shutil.copytree(src,stage/'reused_runs'/identifier)
            reuse.append(dict(id=identifier,model=m,fold=f,seed=5,
                 archive_manifest_sha256=digest(src/'archive_manifest.json'),source_archive=str(src)))
    write(stage/'reused_runs.json',reuse);write(stage/'seed1_controls.json',controls)
    plan=dict(name=NAME,tasks=tasks(),num_runs=6,reused_runs=6,logical_runs=12,budget_flops=CAP,
              models=list(MODELS),folds=[0,1,2],seeds=[5],gpu_workers=2,augmentation='model_specific',
              conventional_python=read(parent/'plan.json')['conventional_python'],graceful_stop_seconds=20700,
              archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
              storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
              authorization='User asks to establish first that substituting seed5 for seed1 works for all models. Reuse six verified completed seed5 baselines; execute six missing SwinUNETR/UNETR fold baselines. Keep all recipes, folds,10 subjects,43T cap fixed. TwoGPU workers, one running and one queued allocation maximum; archive, hash-verify and clean each run.',
              selection_caveat='Development-validation seed selection; passing the mean50% rule does not imply all folds pass or performance improves over seed1. No full14-condition study or stronger-constraint submission in this stage.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
          reused_seed5_manifest_sha256=digest(ROOT/'experiments'/RECOVERY/'manifest.json'),
          changes='Training runtime unchanged. Six missing seed5 baseline runs, six exact verified reuses; seed1 controls preserved.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace('scripts.mni_constraints.shared43_study',MODULE).replace('hippo-mni-14','hippo-mni-seed5'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');parent=read(stage/'parent_manifest.json')
    assert plan['tasks']==tasks() and plan['budget_flops']==CAP and plan['gpu_workers']==2
    allowed={'name','data','seeds','decision_rule','primary_metric'}
    for m in MODELS:
        folder=stage/'protocols'/(m+'_dice');p=read(folder/'protocol.json');old=read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json')==parent[f'protocols/{m}_dice/protocol.json']
        assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
        assert p['seeds']==[5] and p['folds']==[0,1,2] and p['budget_flops']==CAP
        for name,k in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[k]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for group,listing,count in [('reused_runs','reused_runs.json',6),('seed1_controls','seed1_controls.json',12)]:
        records=read(stage/listing);assert len(records)==count
        for item in records:
            folder=stage/group/item['id'];verify_archive(folder,read(folder/'archive_manifest.json'))
            assert digest(folder/'archive_manifest.json')==item['archive_manifest_sha256']
    return plan


def preflight(stage):
    verify(stage)
    import torch,monai
    from scripts.unetr_subfields import core
    from scripts.unetr_subfields.contracts import DATASETS
    torch.set_num_threads(4);assert monai.__version__=='1.6.0'
    frame=core.base._load_pkl_dataframe(stage/'data/MNI.pkl');rows=[]
    for m in MODELS:
        folder=stage/'protocols'/(m+'_dice');p=read(folder/'protocol.json');profile=read(folder/'profile.json')
        batch.bind(SimpleNamespace(protocol_dir=folder,model=p['trainer_alias'],objective='dice',fold=0))
        model=core.base.build_model(p['trainer_alias'],read(folder/'nnunet_architecture.json')).to('meta').train()
        outputs=model(torch.zeros((1,1,64,64,64),device='meta'))
        shapes=[list(x.shape) for x in outputs] if isinstance(outputs,(list,tuple)) else [list(outputs.shape)]
        assert shapes==profile['output_shapes'] and sum(x.numel() for x in model.parameters())==profile['parameters']
        for split in read(folder/'splits_10patients.json'):
            assert len(core.PreparedSubfields(frame,split['train'],DATASETS['MNI']))==20
            assert len(core.PreparedSubfields(frame,split['val'],DATASETS['MNI']))==10
        rows.append(dict(model=m,output_shapes=shapes,updates=p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']))
        del model,outputs
    result=dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=rows,optimizer_updates=0,
                torch=torch.__version__,monai=monai.__version__)
    write(stage/'preflight.json',result);return result


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
