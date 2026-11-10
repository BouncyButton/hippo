"""Create a small frozen deployment; data ZIPs are staged separately."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPORT = ROOT/'reports/sam3_geofwi_f013_s012_126_20261019'
SPLIT = EXPORT/'split_cluster_20261022'
STAGE = SPLIT/'cluster_stage'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(HERE))
    import replication
    state = json.loads((SPLIT/'source_campaign_state.json').read_text())
    assignment = json.loads((SPLIT/'assignment.json').read_text())
    assignment['completed_ids'] = sorted(state['completed'])
    (STAGE/'assignment.json').write_text(json.dumps(assignment,indent=2)+'\n')
    (STAGE/'colab_binding.json').write_text(json.dumps(state['binding'],indent=2)+'\n')
    shutil.copy2(HERE/'cluster_tail_20261022.py',STAGE/'cluster_runner.py')
    shutil.copy2(HERE/'initialize_cluster_tail_20261022.py',STAGE/'initialize.py')
    shutil.copy2(EXPORT/'sam3_126_code.zip',STAGE/'sam3_126_code.zip')
    shutil.copy2(SPLIT/'calibration_f1s2.zip',STAGE/'calibration_f1s2.zip')
    assert digest(STAGE/'calibration_f1s2.zip') == state['objects']['calibration/f1s2']['local_sha256']
    tasks = [dict(t,kind='foundation',dataset='GeoFWI3D',model='SAM3') for t in replication.tasks()
             if t['id'] in assignment['cluster_tasks']]
    plan = dict(name='sam3_f013_s012_cluster_tail_20261022', tasks=tasks,
        num_runs=len(tasks), base_sha256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
        foundation_python='/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928/venv/bin/python',
        workers=2, storage=dict(archive=str(SPLIT/'cluster_results'), minimum_free_gib=12,
            estimated_peak_logical_gib=7, estimated_mirrored_peak_gib=14),
        policy='Frozen Colab scientific bundle; cluster transport, scheduling and paths are audited separately. Every run is verified off BeeGFS before bulky output deletion.')
    (STAGE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    inputs = {p.name:digest(p) for p in EXPORT.glob('sam3_290vol_data12_*.zip')}
    assert len(inputs)==12
    (STAGE/'input_hashes.json').write_text(json.dumps(inputs,indent=2)+'\n')
    shell = '''#!/usr/bin/env bash
#SBATCH --job-name=hippo-sam3-126-tail
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
set -Eeuo pipefail
stage="${1:?Pass the frozen stage}"
export PYTHONPATH="$stage/bundle:$stage/bundle/source:$stage/bundle/source/scripts"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-sam3-tail-mpl-${SLURM_JOB_ID:?}"
cd "$stage"
/mnt/beegfsstudents/home/3160552/medsam3_runtime_20260928/venv/bin/python -u "$stage/cluster_runner.py" run --stage "$stage" &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null || true' USR1 TERM
set +e
wait "$pid"
code=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; code=$?; done
exit "$code"
'''
    (STAGE/'run.sbatch').write_text(shell)
    manifest = {str(p.relative_to(STAGE)):digest(p) for p in STAGE.iterdir() if p.is_file() and p.name!='manifest.json'}
    (STAGE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(STAGE)


if __name__=='__main__': main()
