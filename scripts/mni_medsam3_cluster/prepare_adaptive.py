"""Freeze the60%-Dice,400→1000-update, two-worker MNI adaptive campaign."""
from pathlib import Path
import shutil

from scripts.colab_mni_medsam3.data import select_support
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_medsam3_cluster.prepare import ROOT

NAME='mni_medsam3_n5_adaptive60_2gpu_20261110'
EXPORT=ROOT/'reports'/NAME
STAGE=EXPORT/'stage'
REMOTE=Path('/mnt/beegfsstudents/home/3160552')/NAME
PARENT=ROOT/'reports/mni_medsam3_n5_f0_s0_400_20261110/stage'
MODULE='scripts.mni_medsam3_cluster.adaptive_campaign'


def build():
    if STAGE.exists():raise ValueError('Preserve frozen campaign')
    for n,sha in read(PARENT/'manifest.json').items():
        if digest(PARENT/n)!=sha:raise ValueError('Parent changed: '+n)
    STAGE.mkdir(parents=True)
    for name in ('source','data'):
        shutil.copytree(PARENT/name,STAGE/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('saved_folds.json','upstream_manifest.json','data_audit.json'):
        shutil.copy2(PARENT/name,STAGE/name)
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source,STAGE/'source/scripts/mni_medsam3_cluster'/source.name)
    parent=read(PARENT/'plan.json')
    reference=read(PARENT/'protocols/MNI_MedSAM3_n5_f0s0_native400/protocol.json')
    folds=read(PARENT/'saved_folds.json');tasks=[]
    for fold in (0,1,2):
        selection=select_support(folds,5,fold)
        for seed in (0,1,2):
            for updates in range(400,1001,100):
                identity=f'MNI_MedSAM3_n5_f{fold}s{seed}_native{updates}'
                p=dict(reference,**selection)
                p.update(name=identity,fold=fold,seed=seed,updates=updates)
                # Preserve the initial seed0 protocol exactly, including the
                # immutable dataset path, to resume its in-flight checkpoint.
                if (fold,seed,updates)!=(0,0,400):
                    p.update(sampling=f'{updates} total updates; two uniformly sampled adjacent-slice pairs per update.',
                        support_selection=f'First five subjects of the fixed RandomState(0) fold{fold} training ordering, both L/R crops.',
                        decision_rule='Start400; resume+100 if mean validation subfield Dice<0.60; cap1000. Discover two seeds onfold0, then test them onfolds1/2.')
                # Continuations may change only their update horizon and labels;
                # keep support-selection metadata identical within each seed.
                if fold==0 and seed==0:
                    p['support_selection']=reference['support_selection']
                write(STAGE/'protocols'/identity/'protocol.json',p)
                tasks.append(dict(id=identity,protocol=identity,kind='foundation',dataset='MNI',subjects=5,
                    fold=fold,seed=seed,updates=updates,arm='dice'))
    plan=dict(parent,name=NAME,tasks=tasks,seeds=[0,1,2],threshold=.60,maximum_updates=1000,
        controller_module=MODULE,archive_resume_module=MODULE,
        parallelism=dict(parent['parallelism'],workers=2),graceful_stop_seconds=20700,
        storage=dict(minimum_free_gib=8,estimated_peak_beegfs_gib=4,
            note='Retain one failed milestone checkpoint per active seed only until verified continuation transfer; retire qualified/exhausted checkpoints after archive.'),
        discovery=dict(fold=0,seeds=[0,1,2],required_qualified=2,validation_folds=[1,2]),
        initial_recovery=dict(stage='/mnt/beegfsstudents/home/3160552/mni_medsam3_n5_f0_s0_400_20261110',
            task='MNI_MedSAM3_n5_f0s0_native400',protocol_sha256=digest(PARENT/'protocols/MNI_MedSAM3_n5_f0s0_native400/protocol.json')),
        schedule_policy='Initial400 uses the established400-update cosine schedule. Each failed milestone resumes optimizer/RNG/history and applies the original cosine formula with the expanded horizon to the additional100 updates. No historical updates are replayed.',
        checkpoint_policy='Archive every milestone. Retain checkpoint only for its planned next100-update extension; delete parent after verified child recovery transfer. Qualification or cap ends that dependency.')
    plan.pop('updates_per_run',None);plan.pop('estimated_training_hours',None)
    write(STAGE/'plan.json',plan)
    provenance=read(PARENT/'source_provenance.json')
    provenance.update(parent_manifest_sha256=digest(PARENT/'manifest.json'),
        changes='Adaptive milestone scheduler; exact optimizer/RNG/history continuation with documented horizon updates; two isolatedGPU workers; first two qualifying seeds replicated onfolds1/2.',
        user_threshold=.60,user_maximum_updates=1000,scientific_engine_changed=False)
    write(STAGE/'source_provenance.json',provenance)
    script=(PARENT/'run.sbatch').read_text().replace('hippo-medsam3-n5-400','hippo-medsam3-adaptive')
    script=script.replace('--ntasks=1','--ntasks=2').replace('--gres=gpu:4g.40gb:1','--gres=gpu:4g.40gb:2').replace('--mem=40G','--mem=80G')
    script=script.replace('--time=03:00:00','--time=06:00:00').replace('scripts.mni_medsam3_cluster.campaign run',MODULE+' run')
    (STAGE/'run.sbatch').write_text(script)
    write(STAGE/'manifest.json',{str(p.relative_to(STAGE)):digest(p) for p in sorted(STAGE.rglob('*')) if p.is_file()})
    write(STAGE/'control.json',dict(pause=False))
    write(EXPORT/'preparation.json',dict(status='prepared_not_submitted',stage=str(REMOTE),local_stage=str(STAGE),
        threshold=.60,first_updates=400,increment=100,maximum_updates=1000,seeds=[0,1,2],workers=2,
        policy='Find two qualifying seeds onfold0, then test both onfolds1and2'))
    print(EXPORT)


if __name__=='__main__':build()
