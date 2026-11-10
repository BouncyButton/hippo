"""Freeze the established Colab recipes without changing scientific source."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from .campaign import CAP, MODELS, read, write, sha, verify

ROOT=Path(__file__).resolve().parents[2]
NAME='geofwi3d_frozen_3fold_3seed_20261020'
REMOTE=Path('/mnt/beegfsstudents/home/3160552')/NAME
PARENT=ROOT/'reports/geofwi3d_fold0_baselines_20261019/bundle'
INIT={0:'1215193efc51fc67a9da9cb1ef0bca0e6e8019d4af8146d9efcb85b94bb051f3',
      1:'ef5a5498ba01a17b3f386d284b4b99e6cb599ad3255b06ec85c3f6ff62ab7b88',
      2:'ddcbb9ad986b8d921e0947853c75baaaf83898b04cbc0a218fa92a7af424a653'}


def prepare(fresh=False):
    stage=ROOT/'experiments'/NAME
    if stage.exists():
        plan=verify(stage); assert plan['new_folds']==([0,1,2] if fresh else [1,2]); return stage
    stage.mkdir(parents=True)
    for name, expected in read(PARENT/'manifest.json').items(): assert sha(PARENT/name)==expected,name
    for child in ('unetr','basic'):
        src=PARENT/child; dst=stage/'bundles'/child
        # Copy only the original manifest inventory, not work/checkpoints/caches.
        inventory=read(src/'bundle_manifest.json')
        for name in sorted(set(inventory)|{'bundle_manifest.json'}):
            target=dst/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,target)
        if not (dst/'control_environment.json').exists():
            shutil.copy2(PARENT/'unetr/control_environment.json',dst/'control_environment.json')
    shutil.copy2(PARENT/'basic/splits.json',stage/'splits.json')
    shutil.copy2(ROOT/'scripts/geofwi_cluster/campaign.py',stage/'campaign.py')
    shutil.copy2(ROOT/'scripts/geofwi_cluster/worker_entry.py',stage/'worker_entry.py')
    shutil.copy2(ROOT/'scripts/geofwi_cluster/cpu_audit.py',stage/'cpu_audit.py')
    shutil.copy2(PARENT/'basic/data_manifest.json',stage/'data_manifest.json')
    shutil.copy2(ROOT/'reports/geofwi3d_fold0_baselines_20261019/observed_final_status.json',stage/'colab_fold0_reference.json')
    original_data=(PARENT/'basic/data').resolve()
    (stage/'data').symlink_to(original_data,target_is_directory=True)
    tasks=[]
    prior=read(ROOT/'reports/geofwi3d_unetr_cpu_audit_20261019/search_status_snapshot.json')['runs']
    prior+=read(ROOT/'reports/geofwi3d_fold0_baselines_20261019/observed_basic_final_results.json')
    inits={m:{str(r['seed']):r['initialization_sha256'] for r in prior if r['model']==m and r['fold']==0 and r['seed']<3} for m in MODELS}
    inits['unetr_vit_b16']={str(k):v for k,v in INIT.items()}
    assert all(set(v)=={'0','1','2'} for v in inits.values())
    for fold in ([0,1,2] if fresh else [1,2]):
        for seed in range(3):
            for model in MODELS:
                child='unetr' if model=='unetr_vit_b16' else 'basic'
                source=stage/'bundles'/child/'templates'/model
                p=deepcopy(read(source/'protocol.json'))
                identity=f'GeoFWI3D_{model}_baseline_f{fold}s{seed}'
                folder=stage/'protocols'/identity;folder.mkdir(parents=True)
                remote_folder=REMOTE/'protocols'/identity
                write(folder/'dataset.json',dict(dataset_root=str(REMOTE/'data'),manifest=str(REMOTE/'data/fold_0_manifest.json')))
                shutil.copy2(stage/'splits.json',folder/'splits_10patients.json')
                shutil.copy2(source/'nnunet_architecture.json',folder/'nnunet_architecture.json')
                p.update(data=str(remote_folder/'dataset.json'),pkl_sha256=sha(folder/'dataset.json'),
                    splits_sha256=sha(stage/'splits.json'),folds=[fold],seeds=[seed],baseline_run_id=identity,
                    constraints=False,requested_arm_order=['dice'],objective='dice',reference_initializations=inits[model])
                if child=='unetr':
                    p.update(rescue_arm=f'lr6_f{fold}s{seed}',pilot_initialization_sha256=INIT[seed])
                    s=p['models'][p['trainer_alias']]['objectives']['dice']
                    p['rescue_validation_steps']=sorted({min(s['max_steps'],CAP*i//10//s['flops_per_update']) for i in range(1,11)}|{s['max_steps']})
                write(folder/'protocol.json',p)
                allowed={'data','pkl_sha256','splits_sha256','folds','seeds','baseline_run_id','constraints','requested_arm_order','objective','rescue_arm','pilot_initialization_sha256','rescue_validation_steps','reference_initializations'}
                original=read(source/'protocol.json')
                assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in original.items() if k not in allowed}
                s=p['models'][p['trainer_alias']]['objectives']['dice']
                tasks.append(dict(id=identity,kind='conventional',dataset='GeoFWI3D',model=model,fold=fold,seed=seed,
                    arm='dice',bundle=child,augmentation=p['augmentation'],updates=s['max_steps']))
    references=read(stage/'colab_fold0_reference.json'); assert references['qualified'] and len(references['rows'])==12
    write(stage/'plan.json',dict(name=NAME,workers=2,cap=CAP,new_folds=[0,1,2] if fresh else [1,2],seeds=[0,1,2],
        tasks=tasks,reused_rows=[] if fresh else references['rows'],new_runs=len(tasks),total_cells=36,
        selection_policy='Fixed seeds 0/1/2; retain every result, no replacement seeds, no halfway gate. QUALIFIED only if all36 final Dice strictly exceed .5.',
        hardware_policy='Two isolated A100 MIG 4g.40gb devices; one single-GPU experiment per device, unchanged batch1 and BF16. Colab references came from full A10080GB; provenance remains separate.',
        data_policy='Frozen 19 training and96 validation volumes per fold, native96cubed grid, same three splits.',
        storage=dict(minimum_free_gib=24,estimated_peak_logical_gib=12,mirrored_peak_allowance_gib=24,
            archive=str(ROOT/'reports'/NAME),per_run_verified_archive_before_retirement=True),
        gpu_preflight='Three diagnostic updates per model per lane/allocation, separately recorded outside production budget.',
        parent_manifest_sha256=sha(PARENT/'manifest.json')))
    batch='''#!/usr/bin/env bash
#SBATCH --job-name=hippo-geofwi-3f3s
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
set -Eeuo pipefail
stage="${1:?Pass frozen stage}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$stage"
/home/3160552/.conda/envs/hippocampus/bin/python -u "$stage/campaign.py" run --stage "$stage" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1 TERM
set +e
wait "$pid"
code=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; code=$?; done
exit "$code"
'''
    (stage/'run.sbatch').write_text(batch)
    write(stage/'manifest.json',{str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file() and 'data' not in p.relative_to(stage).parts})
    verify(stage)
    report=ROOT/'reports'/NAME;report.mkdir(parents=True,exist_ok=True)
    shutil.copytree(stage,report/'frozen',symlinks=True)
    write(report/'preparation.json',dict(status='prepared',new_runs=len(tasks),reused=0 if fresh else 12,manifest_sha256=sha(stage/'manifest.json')))
    return stage


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');args=ap.parse_args();print(prepare(args.fresh))
