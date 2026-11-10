#!/usr/bin/env bash
# Prepared only. Submit manually when the one-running/one-queued policy permits.
#SBATCH --job-name=hippo-cap-replication
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${HIPPO_REPLICATION_STAGE:-${SLURM_SUBMIT_DIR:?}}"
root="/mnt/beegfsstudents/home/3160552/capped_replication_31976_20261007_results"
python="/home/3160552/.conda/envs/hippocampus/bin/python"
dataset="/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
[[ -f "${stage}/source_manifest.json" && -f "${stage}/source/scripts/capped_replication_worker.py" ]]
export PYTHONPATH="${stage}/source:${stage}/source/scripts:${stage}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-replication-mpl-${SLURM_JOB_ID:?}"
"${python}" -m scripts.capped_replication_worker --stage "${stage}" --root "${root}" \
  --pkl "${dataset}" --safe-runtime-seconds 85200 &
worker_pid=$!
trap 'kill -USR1 "${worker_pid}" 2>/dev/null || true' USR1
trap 'kill -TERM "${worker_pid}" 2>/dev/null || true' TERM INT
status=0
while kill -0 "${worker_pid}" 2>/dev/null; do
  set +e
  wait "${worker_pid}"
  status=$?
  set -e
  if (( status <= 128 )); then
    break
  fi
done
exit "${status}"
