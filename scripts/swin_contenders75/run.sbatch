#!/usr/bin/env bash
#SBATCH --job-name=hippo-swin-contenders75
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/swin_contenders75_20261013_results"
python="/home/3160552/.conda/envs/hippocampus/bin/python"
export PYTHONPATH="${stage}/source:${stage}/source/scripts:${stage}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-contenders-mpl-${SLURM_JOB_ID:?}"
"${python}" -m scripts.swin_contenders75.worker --stage "${stage}" --root "${root}" &
worker_pid=$!
trap 'kill -USR1 "${worker_pid}" 2>/dev/null || true' USR1
trap 'kill -TERM "${worker_pid}" 2>/dev/null || true' TERM INT
status=0
while kill -0 "${worker_pid}" 2>/dev/null; do
  set +e
  wait "${worker_pid}"
  status=$?
  set -e
  if (( status <= 128 )); then break; fi
done
exit "${status}"
