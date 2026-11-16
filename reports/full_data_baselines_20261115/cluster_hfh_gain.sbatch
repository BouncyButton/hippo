#!/usr/bin/env bash
#SBATCH --job-name=HFH-gain-match
#SBATCH --partition=stud
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=gnode04
set -Eeuo pipefail
base=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
cd "$base/full_data_baselines_20261115/cluster_hfh_gain"
original="$base/hfh_medsam3_seed2_20261114/bundle"
export PYTHONPATH="$original/source:$original/source/scripts:$original/source/baselines/nnUNet"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HFH_CONTROLLER_ARCHIVES=1
export HFH_BASE_WEIGHTS="$base/sam3_f013_s012_cluster_tail_20261022/bundle/weights/sam3.pt"
export TORCHINDUCTOR_CACHE_DIR="/tmp/hippo-hfh-gain-$UID/inductor" TRITON_CACHE_DIR="/tmp/hippo-hfh-gain-$UID/triton" MPLCONFIGDIR="/tmp/hippo-hfh-gain-$UID/mpl"
export MNI_MANIFEST_SHA="$(sha256sum SHA256SUMS | cut -d' ' -f1)"
py="$base/medsam3_runtime_20260928/venv/bin/python"
sha256sum --status -c SHA256SUMS
"$py" cluster_hfh_gain.py preflight
specs=(protocols/HFH_*.json)
test "${#specs[@]}" -eq 9
lane() (
    for ((i=$1; i<9; i+=2)); do
        spec="${specs[$i]}"; key="$(basename "$spec" .json)"; out="$PWD/work/$key"
        [[ -f "$out/cleanup.json" ]] && continue
        mkdir -p "$out"
        "$py" -c 'from scripts.cross_dataset_campaign.storage import quota; assert quota()["free_gib"] >= 12'
        if [[ ! -f "$out/completion.json" ]]; then
            srun --exclusive --exact -N1 -n1 -c4 --mem=40G --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$py" -u -m scripts.colab_hfh.medsam_replication_matrix_worker --protocol "$PWD/$spec" --output "$out" --durable "$PWD/recovery/$key" --base-weights "$HFH_BASE_WEIGHTS" --arm dice >>"$out/training.log" 2>&1
        fi
        "$py" cluster_hfh_gain.py archive "$key"
        for ((j=0; j<120; j++)); do [[ -f "$out/cleanup.json" ]] && break; sleep 5; done
        [[ -f "$out/cleanup.json" ]] || exit 75
    done
)
lane 0 & first=$!
lane 1 & second=$!
status=0
wait "$first" || status=$?
wait "$second" || status=$?
exit "$status"
