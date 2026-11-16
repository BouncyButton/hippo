#!/usr/bin/env bash
#SBATCH --job-name=full-MNI-med-pair
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode04
set -Eeuo pipefail
root=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
stage="$root/full_data_baselines_20261115"
parent="$root/mni_medsam3_n5_f0_s0_400_20261110"
python="$root/medsam3_runtime_20260928/venv/bin/python"
cd "$stage/mni_adapter"
sha256sum --status -c SHA256SUMS
cd "$parent"
sha256sum --status -c "$stage/parent_inputs.sha256"
export PYTHONPATH="$parent/source:$parent/source/scripts:$parent/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-full-mpl-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="/tmp/hippo-full-compiler-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
export MNI_MANIFEST_SHA="$(sha256sum "$stage/mni_adapter/SHA256SUMS" | cut -d' ' -f1)"
cd "$parent/source"
test "$#" -eq 2
test "$1" != "$2"
for key in "$@"; do
    [[ "$key" =~ ^MNI_medsam3_f[012]_s[012]$ ]]
    test -f "$stage/mni_adapter/protocols/$key.json"
    test ! -f "$stage/runs/$key/completion.json"
done
"$python" -c 'from scripts.cross_dataset_campaign.storage import quota; q=quota(); print(q,flush=True); assert q["free_gib"] >= 12'
run_cell() (
    key="$1"
    export HIPPO_GPU_WORKER="$2"
    output="$stage/runs/$key"
    mkdir -p "$output"
    exec 9>"$output/worker.lock"
    flock -n 9
    srun --exclusive --exact -N1 -n1 --cpus-per-task=4 --mem=40G --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$python" -u -m scripts.mni_medsam3_cluster.worker --protocol "$stage/mni_adapter/protocols/$key.json" --output "$output" --arm dice --base-weights "$root/sam3_f013_s012_cluster_tail_20261022/bundle/weights/sam3.pt" --record "$output/gpu_worker.json" >> "$output/training.log" 2>&1
)
run_cell "$1" 0 &
first=$!
run_cell "$2" 1 &
second=$!
status=0
wait "$first" || status=$?
wait "$second" || status=$?
exit "$status"
