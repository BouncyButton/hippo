"""Verify the local backup and prepare an isolated recovery control bundle."""
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from scripts.adni_constraints.recovery_worker import digest,read,write,verify_completed,replay_ledger

NAME='adni_constraints_noaug_recovery_20261013'
ORIGINAL='adni_constraints_noaug_5pct_fold0_seed0_20261013'


def main():
    local=ROOT/'experiments'/NAME
    backup=local/'backup'/(ORIGINAL+'_results')
    frozen=ROOT/'experiments'/ORIGINAL
    report=ROOT/'reports'/ORIGINAL/'recovery_20261013'
    stage=local/'control'
    if stage.exists():
        raise FileExistsError('Do not overwrite a frozen recovery')
    plan,summary=read(frozen/'plan.json'),read(backup/'summary.json')
    if summary['completed_runs']!=22 or summary['augmentation']!='none':
        raise ValueError('Unexpected backup state')
    completed={(r['model'],r['objective']) for r in summary['runs']}
    files=[]
    for run in plan['runs']:
        if (run['model'],run['objective']) not in completed:
            continue
        verify_completed(frozen,backup,run,{})
        relative=Path(run['model'])/run['objective']/'fold0/seed0/run'
        checkpoint=backup/relative/'checkpoint_best.pt'
        files.append(dict(relative_path=str(relative/'checkpoint_best.pt'),sha256=digest(checkpoint),
            bytes=checkpoint.stat().st_size,local_backup=str(checkpoint),
            completion_sha256=digest(backup/relative/'completion.json')))
    interrupted=Path('unetr_vit_b16/dice_edge_depth/fold0/seed0/run')
    source=backup/interrupted
    if not zipfile.is_zipfile(source/'checkpoint_latest.pt'):
        raise ValueError('Local resume checkpoint container is incomplete')
    inspection=read(report/'checkpoint_inspection.json')
    if inspection['checkpoint_step']!=147 or inspection['checkpoint_charged_attempts']!=147:
        raise ValueError('Unexpected inspected checkpoint state')
    if inspection['original_ledger']!=read(source/'budget_ledger.json'):
        raise ValueError('Backup ledger differs from the inspected interrupted run')
    protocol=read(frozen/'protocols/unetr_vit_b16/protocol.json')
    schedule=protocol['models']['unet']['objectives']['dice_edge_depth']
    _,replay=replay_ledger(read(source/'budget_ledger.json'),147,schedule)
    stage.mkdir(parents=True)
    archive=dict(status='verified_local_backups',verified_at_unix=time.time(),files=files,
                 completed_runs=22,total_bytes=sum(f['bytes'] for f in files),backup_root=str(backup))
    write(stage/'checkpoint_archive.json',archive)
    remote='/mnt/beegfsstudents/home/3160552/'
    remaining=[r for r in plan['runs'] if (r['model'],r['objective']) not in completed]
    params={m:read(frozen/'protocols'/m/'protocol.json')['models'][next(r['alias'] for r in plan['runs'] if r['model']==m)]['parameters'] for m in plan['models']}
    remaining_bytes=sum(4*params[r['model']] for r in remaining)
    control=dict(name=NAME,original_stage=remote+ORIGINAL,result_root=remote+ORIGINAL+'_results',
        original_manifest_sha256=digest(frozen/'manifest.json'),original_failed_job=698935,
        completed_runs_reused=22,unfinished_runs=34,never_started_runs=33,augmentation='none',fold=0,seed=0,
        interrupted_relative_path=str(interrupted),
        interrupted_archive_relative_path=str(Path('_interrupted_job698935')/interrupted),
        checkpoint_sha256=digest(source/'checkpoint_latest.pt'),checkpoint_bytes=(source/'checkpoint_latest.pt').stat().st_size,
        interrupted_metadata_sha256={f.name:digest(f) for f in source.iterdir() if f.is_file() and f.suffix in ('.json','.csv')},
        replay=replay,training_cap_flops_per_completed_trajectory=plan['budget_flops_per_run'],
        minimum_quota_free_gib=2*remaining_bytes/(1024**3)+2,
        quota_policy='Reserve twice the remaining retained-model file bytes plus 2 GiB; transient optimizer writes use node-local scratch.',
        authorization='User explicitly requested freeing necessary space and resuming after disclosure of possible extra recovery compute.',
        checkpoint_retention='22 completed best models preserved in verified local backups; future best models retained on cluster. Interrupted checkpoint and original ledger retained separately.',
        accounting='Physical compute for the resumed condition equals its restored trajectory plus explicitly recorded replay overhead. Calibration is retained, not repeated.')
    write(stage/'recovery_plan.json',control)
    shutil.copyfile(ROOT/'scripts/adni_constraints/recovery_worker.py',stage/'worker.py')
    shutil.copyfile(ROOT/'scripts/adni_constraints/recovery_submit.py',stage/'submit.py')
    shutil.copyfile(ROOT/'scripts/adni_constraints/recovery_storage.py',stage/'storage.py')
    script='''#!/usr/bin/env bash
#SBATCH --job-name=hippo-adni-noaug-recovery
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
recovery="${SLURM_SUBMIT_DIR:?}"
source="/mnt/beegfsstudents/home/3160552/adni_constraints_noaug_5pct_fold0_seed0_20261013/source"
export PYTHONPATH="${source}:${source}/scripts:${source}/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-adni-noaug-recovery-mpl-${SLURM_JOB_ID:?}"
/home/3160552/.conda/envs/hippocampus/bin/python "${recovery}/worker.py" --recovery "${recovery}"
'''
    (stage/'run.sbatch').write_text(script)
    write(stage/'manifest.json',{p.name:digest(p) for p in sorted(stage.iterdir()) if p.is_file()})
    write(report/'recovery_plan.json',control)
    write(report/'checkpoint_archive.json',archive)
    write(report/'preflight.json',dict(status='passed',verified_completed_runs=22,
        intact_checkpoint_update=147,original_charged_update=159,replay=replay,
        all_completed_metadata_and_best_models_verified=True,manifest_sha256=digest(stage/'manifest.json')))
    with tarfile.open(report/'recovery_bundle.tar.gz','w:gz') as bundle:
        for p in sorted(stage.iterdir()):
            bundle.add(p,arcname=p.name,recursive=False)
    print(json.dumps(dict(status='ready',archived_checkpoint_bytes=archive['total_bytes'],
                         remaining_runs=34,replay=replay,required_quota_free_gib=control['minimum_quota_free_gib'])),flush=True)


if __name__=='__main__':
    main()
