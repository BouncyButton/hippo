#!/usr/bin/env bash
#SBATCH --job-name=full-ADNI-adapter
#SBATCH --partition=stud
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=gnode04
set -Eeuo pipefail
root=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
cd "$root/full_data_baselines_20261115/cluster_medical"
sha256sum --status -c SHA256SUMS
py="$root/medsam3_runtime_20260928/venv/bin/python"
export PYTHONPATH="$PWD/source:$PWD/source/scripts:$PWD/source/baselines/nnUNet"
export MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MNI_MANIFEST_SHA="$(sha256sum SHA256SUMS | cut -d' ' -f1)"
export MPLCONFIGDIR="/tmp/hippo-full-adni-mpl-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="/tmp/hippo-full-adni-compiler-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR/triton"
lane() (
    local lane="$1"; shift
    export HIPPO_GPU_WORKER="$lane"
    for key in "$@"; do
        out="work/$key"
        mkdir -p "$out"
        if [[ -f "$out/cleanup.json" ]]; then continue; fi
        "$py" -c 'from scripts.cross_dataset_campaign.storage import quota; assert quota()["free_gib"] >= 12'
        if [[ ! -f "$out/completion.json" ]]; then
            srun --exclusive --exact -N1 -n1 --cpus-per-task=4 --mem=40G --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$py" -u cluster_medical.py train "$key" >> "$out/training.log" 2>&1
        fi
        "$py" cluster_medical.py archive "$key"
        for ((i=0; i<120; i++)); do
            [[ -f "$out/cleanup.json" ]] && break
            sleep 5
        done
        [[ -f "$out/cleanup.json" ]] || exit 75
        "$py" cluster_medical.py release-local "$key"
    done
)
lane 0 ADNI_medsam3_f3_s0 ADNI_medsam3_f1_s2 &
first=$!
lane 1 ADNI_medsam3_f3_s1 ADNI_medsam3_f3_s2 &
second=$!
status=0
wait "$first" || status=$?
wait "$second" || status=$?
exit "$status"
