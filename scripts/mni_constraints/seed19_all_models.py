"""Validate common seed19 on all models before matched UNETR constraint work."""
import shutil
from pathlib import Path
from scripts.cross_dataset_campaign.common import read,write,digest,REMOTE
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.mni_constraints import campaign,headroom_bands,seed5_all_models as common
from scripts.mni_constraints.shared43_study import ROOT,NAME as PARENT,CAP,MODELS

NAME='mni_seed19_all_models_shared43t_20261108'
MODULE='scripts.mni_constraints.seed19_all_models'
PREVIOUS='mni_unet_foldscreen_shared43t_20261108'


def tasks():
    return [dict(id=f'MNI_{m}_dice_f{f}s19',kind='conventional',dataset='MNI',model=m,
        arm='dice',fold=f,seed=19,protocol=m+'_dice',augmentation='none' if m=='nnunet' else 'xy_d4_v1')
        for f in range(3) for m in ('nnunet','swinunetr','unetr_vit_b16')]


def prepare(_target=None):
    stage=ROOT/'experiments'/NAME;parent=ROOT/'experiments'/PARENT
    if (stage/'manifest.json').exists():return verify(stage)
    for name,sha in read(parent/'manifest.json').items():assert digest(parent/name)==sha,name
    for part in ('source','data'):
        shutil.copytree(parent/part,stage/part,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('seed19_all_models.py','collect_seed19_all_models.py','seed5_all_models.py'):
        shutil.copy2(ROOT/'scripts/mni_constraints'/name,stage/'source/scripts/mni_constraints'/name)
    shutil.copy2(parent/'manifest.json',stage/'parent_manifest.json')
    reuse=[];controls=[];donors={}
    def archive(src,dst,remote_src):
        verify_archive(src,read(src/'archive_manifest.json'))
        assert read(src/'cleanup.json')['status']=='complete'
        shutil.copytree(src,dst)
        for f in dst.iterdir():
            if f.is_file() and f.name!='archive_acknowledgment.json':
                donors[str(f.relative_to(stage))]=str(remote_src/f.name)
    for model in MODELS:
        folder=stage/'protocols'/(model+'_dice');shutil.copytree(parent/'protocols'/(model+'_dice'),folder)
        shutil.copy2(folder/'protocol.json',folder/'parent_protocol.json');p=read(folder/'protocol.json')
        p.update(name=NAME,data=str(REMOTE/NAME/'data/MNI.pkl'),seeds=[19],
            decision_rule='Extend verified UNet seed19 to all models on all original folds. Report every final fold, mean50 and every-fold50. Only proceed to common-seed UNETR constraint experiments if seed19 passes every fold for all models.',
            primary_metric='Final-budget validation foreground macro Dice, fixed original folds0,1,2.')
        write(folder/'protocol.json',p)
        for seed in (0,5):
            for fold in range(3):
                identifier=f'MNI_{model}_dice_f{fold}s{seed}'
                src=ROOT/'reports'/PREVIOUS/'frozen/known_runs'/identifier
                archive(src,stage/'controls'/identifier,REMOTE/PREVIOUS/'known_runs'/identifier)
                controls.append(dict(id=identifier,model=model,seed=seed,fold=fold,
                    archive_manifest_sha256=digest(src/'archive_manifest.json')))
        if model!='basic_unet16':continue
        for fold in range(3):
            identifier=f'MNI_{model}_dice_f{fold}s19';src=ROOT/'reports'/PREVIOUS/'runs'/identifier
            cfg=read(src/'config.json');split=read(folder/'splits_10patients.json')[fold]
            assert cfg['schedule']==p['models'][p['trainer_alias']]['objectives']['dice']
            assert cfg['initial_lr']==p['initial_lrs'][p['trainer_alias']] and cfg['lr_horizon_updates']==p['lr_horizon_updates']
            assert cfg['augmentation']==p['augmentation'] and cfg['budget_flops']==CAP
            assert cfg['train_cases']==split['train'] and cfg['validation_cases']==split['val']
            assert cfg['dataset_sha256']==digest(stage/'data/MNI.pkl')
            assert cfg['script_sha256']==digest(stage/'source/scripts/cross_dataset_campaign/capped_core.py')
            archive(src,stage/'reused_runs'/identifier,REMOTE/PREVIOUS/'runs'/identifier)
            reuse.append(dict(id=identifier,model=model,seed=19,fold=fold,
                archive_manifest_sha256=digest(src/'archive_manifest.json')))
    write(stage/'reused_runs.json',reuse);write(stage/'controls.json',controls)
    write(stage/'deployment_known_donors.json',donors)
    plan=dict(name=NAME,tasks=tasks(),num_runs=9,reused_runs=3,logical_runs=12,budget_flops=CAP,
        models=list(MODELS),folds=[0,1,2],seeds=[19],candidate_common_seeds=[0,5,19],
        gpu_workers=2,augmentation='model_specific',conventional_python=read(parent/'plan.json')['conventional_python'],
        graceful_stop_seconds=20700,archive_wait_timeout_seconds=600,archive_resume_module=MODULE,
        storage=dict(minimum_free_gib=16,estimated_peak_mirrored_gib=12,archive=str(ROOT/'reports'/NAME)),
        authorization='User: extend seed19 to all models; if it works, improve UNETR constrained performance. Reuse three verified UNet runs, execute nine missing baselines with unchanged recipes/data/folds/43T cap. Two GPU workers, one running/one queued allocation; per-run hash-verified archive and cleanup.',
        selection_caveat='Development-validation seed selection. Existing UNet seed5 remains a mean-only accepted seed. This phase checks seed19 on all models; subsequent constraint work requires its all-fold50 gate.')
    write(stage/'plan.json',plan);write(stage/'control.json',dict(pause=False))
    write(stage/'source_version.json',dict(parent_manifest_sha256=digest(parent/'manifest.json'),
        recovered_seed19_manifest_sha256=digest(ROOT/'experiments'/PREVIOUS/'manifest.json'),
        changes='Training runtime unchanged; nine missing seed19 baselines and three verified reuses. Freeze seed0/5 references for the proposed shared seed set.'))
    (stage/'run.sbatch').write_text((parent/'run.sbatch').read_text().replace(
        'scripts.mni_constraints.shared43_study',MODULE).replace('hippo-mni-14','hippo-mni-seed19'))
    campaign.freeze(stage);return verify(stage)


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():assert digest(stage/name)==sha,name
    plan=read(stage/'plan.json');parent=read(stage/'parent_manifest.json')
    assert plan['tasks']==tasks() and plan['budget_flops']==CAP and plan['gpu_workers']==2
    allowed={'name','data','seeds','decision_rule','primary_metric'}
    for model in MODELS:
        folder=stage/'protocols'/(model+'_dice');p=read(folder/'protocol.json');old=read(folder/'parent_protocol.json')
        assert digest(folder/'parent_protocol.json')==parent[f'protocols/{model}_dice/protocol.json']
        assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
        assert p['seeds']==[19] and p['folds']==[0,1,2] and p['budget_flops']==CAP
        for name,key in [('profile.json','profile_sha256'),('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256')]:assert digest(folder/name)==p[key]
        assert digest(stage/'data/MNI.pkl')==p['pkl_sha256']
    for name,sha in parent.items():
        if name.startswith('source/') and name!='source/source_manifest.json':assert digest(stage/name)==sha,name
    for group,listing,count in [('reused_runs','reused_runs.json',3),('controls','controls.json',24)]:
        records=read(stage/listing);assert len(records)==count
        for item in records:
            d=stage/group/item['id'];verify_archive(d,read(d/'archive_manifest.json'))
            assert digest(d/'archive_manifest.json')==item['archive_manifest_sha256']
    return plan


def preflight(stage):
    common.verify=verify
    return common.preflight(stage)


def main():
    campaign.MODULE=MODULE;campaign.prepare=prepare;campaign.verify=verify
    campaign.preflight=preflight;campaign.train=headroom_bands.train
    campaign.main()


if __name__=='__main__':main()
