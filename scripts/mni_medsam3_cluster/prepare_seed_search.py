"""Freeze the authorized sequential third-seed extension without changing live work."""
import os
from pathlib import Path
import shutil
from scripts.cross_dataset_campaign.common import read,write,digest

ROOT=Path(__file__).resolve().parents[2]
NAME='mni_medsam3_n5_third_seed_search_20261110'
REMOTE=(Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())))
PRIOR='mni_medsam3_n5_f012_s01_constraints_20261110'


def build():
    prior=ROOT/'reports'/PRIOR/'stage';stage=ROOT/'reports'/NAME/'stage'
    if stage.exists():raise ValueError('Existing frozen search must be preserved')
    for name,sha in read(prior/'manifest.json').items():
        if digest(prior/name)!=sha:raise ValueError('Parent source changed: '+name)
    stage.mkdir(parents=True)
    shutil.copytree(prior/'source',stage/'source',ignore=shutil.ignore_patterns('__pycache__'))
    additions=['constraint_campaign.py','test_constraint_campaign.py','seed_search.py','test_seed_search.py',
               'prepare_seed_search.py','seed_search_mirror.py','offline_ops/offline_ops.py']
    for name in additions:
        shutil.copy2(Path(__file__).parent/name,stage/'source/scripts/mni_medsam3_cluster'/name)
    old=read(prior/'plan.json')
    plan=dict(name=NAME,previous_matrix_stage=str(REMOTE/PRIOR),previous_matrix_manifest_sha256=digest(prior/'manifest.json'),
        baseline_stage=old['baseline_stage'],baseline_manifest_sha256=old['baseline_manifest_sha256'],
        base_source=old['base_source'],base_sha256=old['base_sha256'],conventional_python=old['conventional_python'],
        foundation_python=old['foundation_python'],candidate_seeds='2,3,4,... in order until one candidate completes the gated full matrix',
        fixed_seeds=[0,1],folds=[0,1,2],subjects_per_fold=5,workers=2,
        baseline_rule='All3 folds >=0.60; start400 and resume in100-update steps, cap1000 per fold per seed.',
        pilot_rule='Run both bands and bands+edge on all3 folds. Continue if either mean paired Dice delta>0; otherwise try next seed.',
        ordering='Finish all78 current constraint runs first. No candidate GPU work before previous matrix completes.',
        final_analysis='14 conditions x3 folds x3 selected seeds =126 cells, with all rejected candidate evidence retained.',
        continuation='Cluster login service delivers weights and resubmits after each6h allocation; USR1 and checkpoint at5h45; no laptop dependency.',
        queue_policy='At most one running and one pending account job.',
        storage=dict(minimum_free_gib=8,estimated_additional_peak_gib_per_candidate=14,
            estimated_remaining_existing_matrix_gib=17,note='Bound includes up to21 baseline milestone checkpoints,39 constraint checkpoints,temporary copies. Retire only after verified off-BeeGFS archive; quota gates remain enabled.'),
        source_additions=additions)
    write(stage/'plan.json',plan)
    (stage/'run.sbatch').write_text((prior/'run.sbatch').read_text())
    unchanged={name:sha for name,sha in read(prior/'manifest.json').items() if name.startswith('source/')
        and name not in {'source/scripts/mni_medsam3_cluster/'+n for n in additions}}
    for name,sha in unchanged.items():
        if digest(stage/name)!=sha:raise ValueError('Scientific engine changed: '+name)
    write(stage/'source_provenance.json',dict(parent_manifest_sha256=digest(prior/'manifest.json'),
        unchanged_reference_files=unchanged,changes=additions))
    write(stage/'manifest.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    write(stage/'control.json',dict(pause=False))
    write(stage.parent/'preparation.json',dict(status='prepared_not_deployed',stage=str(REMOTE/NAME),
        local_stage=str(stage),manifest_sha256=digest(stage/'manifest.json')))
    print(stage)


if __name__=='__main__':build()
