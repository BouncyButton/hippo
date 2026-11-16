#!/bin/bash
#SBATCH --job-name=full-lr-v1
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=gnode04
set -euo pipefail
base=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
here="$base/full_data_baselines_20261115/full_lr_v1"
cd "$here"
sha256sum --status -c SHA256SUMS
py="$base/.conda/envs/hippocampus/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$base/mni_medsam3_n5_f0_s0_400_20261110/source"
export TORCHINDUCTOR_CACHE_DIR="/tmp/hippo-lr-v1-$UID/inductor" TRITON_CACHE_DIR="/tmp/hippo-lr-v1-$UID/triton" MPLCONFIGDIR="/tmp/hippo-lr-v1-$UID/mpl"
mapfile -t keys < <("$py" -c "import json; p=json.load(open('plan.json'));print('\n'.join(sorted([k for k,r in p.items() if r['status']=='new' and (r['dataset']=='MNI' or (r['dataset']=='HFH' and r['model']=='medsam3'))],key=lambda k: ('medsam3' not in k,k))))")
test "${#keys[@]}" -eq 54
lane() (
  for ((i=$1;i<54;i+=2));do
    k="${keys[$i]}"; ds="${k%%_*}"; out="$ds/work/${k}_lr2x"
    [[ -f "$out/cleanup.json" ]] && continue
    "$py" -c 'from scripts.cross_dataset_campaign.storage import quota;assert quota()["free_gib"]>=12'
    srun --exact --exclusive -N1 -n1 -c4 --mem=40G --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$py" -u run.py "$k"
    for ((j=0;j<120;j++));do [[ -f "$out/cleanup.json" ]] && break;sleep 5;done
    [[ -f "$out/cleanup.json" ]] || exit 75
  done
)
lane 0 & first=$!
lane 1 & second=$!
rc=0
wait "$first" || rc=$?
wait "$second" || rc=$?
exit "$rc"
