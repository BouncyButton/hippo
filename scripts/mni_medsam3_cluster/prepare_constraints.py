"""Freeze the user-authorized baseline -> bands/edge -> full matrix chain."""
from pathlib import Path
import shutil
from scripts.cross_dataset_campaign.common import ARMS, read, write, digest

ROOT=Path(__file__).resolve().parents[2]
NAME='mni_medsam3_n5_f012_s01_constraints_20261110'
PARENT_NAME='mni_medsam3_n5_adaptive60_2gpu_20261110'
REMOTE=Path('/mnt/beegfsstudents/home/3160552')
ADDITIONS=('constraint_campaign.py','constraint_worker.py','constraint_support.py',
           'constraint_mirror.py','test_constraint_campaign.py','prepare_constraints.py')


def build():
    parent=ROOT/'reports'/PARENT_NAME/'stage'
    stage=ROOT/'reports'/NAME/'stage'
    if stage.exists():raise ValueError('Preserve existing frozen stage')
    for name,sha in read(parent/'manifest.json').items():
        if digest(parent/name)!=sha:raise ValueError('Parent source changed: '+name)
    stage.mkdir(parents=True)
    shutil.copytree(parent/'source',stage/'source',ignore=shutil.ignore_patterns('__pycache__'))
    destination=stage/'source/scripts/mni_medsam3_cluster'
    for name in ADDITIONS:shutil.copy2(Path(__file__).parent/name,destination/name)
    shutil.copytree(Path(__file__).parent/'offline_ops',destination/'offline_ops',ignore=shutil.ignore_patterns('__pycache__'))
    old=read(parent/'plan.json')
    config=read(ROOT/'reports'/PARENT_NAME/'operations/offline_v1/config.json')
    plan=dict(name=NAME,baseline_stage=str(REMOTE/PARENT_NAME),
        baseline_manifest_sha256=digest(parent/'manifest.json'),
        baseline_controller_sha256=digest(ROOT/'reports'/PARENT_NAME/'operations/offline_v1/offline_ops.py'),
        seeds=[0,1],folds=[0,1,2],subjects_per_fold=5,threshold=.60,baseline_maximum_updates=1000,
        arms=list(ARMS),pilot_arms=['dice_bands','dice_bands_edge'],tasks=[],new_training_runs=78,
        base_source=config['base_source'],base_staging=config['base_source'],base_sha256=old['base_sha256'],
        foundation_python=old['foundation_python'],conventional_python=old['conventional_python'],
        parallelism=old['parallelism'],graceful_stop_seconds=20700,max_job_hours=6,
        archive_wait_timeout_seconds=300,storage=dict(minimum_free_gib=8),offline_peak_gib=26,
        stages=['All six chosen baseline pairs >=60% Dice','12 bands/bands+edge paired runs',
                'Paired pilot report, then remaining 66 constraints regardless of effect direction'],
        matching='Same five subjects, medical initialization, sample sequence and recorded LR history as each first qualifying baseline. Fresh initialization for every arm.',
        calibration='Original ADNI full-matrix sequential training-only gradient calibration: median target .1, p95 cap .5, 4 slabs per crop, constraint warmup20.',
        source_policy='Native trainer, loss functions, geometry algorithm and evaluation acceleration copied unchanged from the frozen baseline source. Separate MNI foreground-union depth adapter preserves all subfield supervision.',
        storage_policy='Retain artifacts on BeeGFS while laptop offline; retire only after verified durable local archive. Estimated 26GiB includes worst-case remaining baseline milestones,78 matrix checkpoints and temporary copies.',
        queue_policy='At most one running and one pending account job; independent GPU workers; cluster service owns delivery and continuation.')
    write(stage/'plan.json',plan)
    script=(parent/'run.sbatch').read_text().replace('hippo-medsam3-adaptive','hippo-medsam3-mni-matrix')
    script=script.replace('scripts.mni_medsam3_cluster.adaptive_campaign','scripts.mni_medsam3_cluster.constraint_campaign')
    (stage/'run.sbatch').write_text(script)
    source_paths=['scripts/cross_dataset_campaign/'+n for n in ('foundation.py','foundation_msd.py','adni_foundation_full_runtime.py')]
    for name in source_paths:
        if digest(stage/'source'/name)!=digest(parent/'source'/name):raise ValueError('Reference engine changed')
    write(stage/'source_provenance.json',dict(parent_manifest_sha256=plan['baseline_manifest_sha256'],
        unchanged_reference_sources={name:digest(stage/'source'/name) for name in source_paths},
        additions=list(ADDITIONS),cohort_resolution='Derived protocols are bound to verified baseline completion/protocol hashes and archived within each run.'))
    write(stage/'manifest.json',{str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    write(stage/'control.json',dict(pause=False,controller_owner='offline_v1',automatic_continuation_owner='cluster_matrix_chain'))
    write(stage.parent/'preparation.json',dict(stage=str(REMOTE/NAME),local_stage=str(stage),new_runs=78,
        existing_baselines=6,workers=2,manifest_sha256=digest(stage/'manifest.json'),status='prepared_not_submitted'))
    print(stage)


if __name__=='__main__':build()
