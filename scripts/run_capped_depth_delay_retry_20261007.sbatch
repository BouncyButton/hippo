#!/usr/bin/env bash
# Same frozen 44 training conditions; exports are verified on the cluster.
#SBATCH --job-name=hippo-cap-depth-retry
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/capped_depth_delay_31976_${SLURM_JOB_ID:?}"
mkdir -p "${root}/operations"
cp -r "${stage}/source" "${root}/source"
cp "${stage}/source_manifest.json" "${root}/source_manifest.json"
cp "${stage}/preflight.json" "${root}/preflight.json"
cp "${stage}/capped_depth_delay_store.py" "${root}/operations/"
cp "${stage}/run_capped_depth_delay_retry_20261007.sbatch" "${root}/operations/"
cp "${stage}/recovery_operations.json" "${root}/operations/"
cmp "${stage}/source_manifest.json" "${root}/source_manifest.json"
export PYTHONPATH="${root}/source:${root}/source/scripts:${root}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-cap-depth-retry-mpl-${SLURM_JOB_ID}"
python3 "${root}/operations/capped_depth_delay_store.py" --root "${root}" &
store_pid=$!
/home/3160552/.conda/envs/hippocampus/bin/python -m scripts.capped_depth_delay.worker \
  --root "${root}" --scratch "/tmp/hippo-capped-depth-delay-${UID}-${SLURM_JOB_ID}" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl &
worker_pid=$!
trap 'kill "${store_pid}" "${worker_pid}" 2>/dev/null || true' EXIT
set +e
wait -n "${worker_pid}" "${store_pid}"
first_status=$?
set -e
if (( first_status != 0 )); then
  exit "${first_status}"
fi
wait "${worker_pid}"
wait "${store_pid}"
