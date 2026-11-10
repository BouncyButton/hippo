#!/usr/bin/env bash
#SBATCH --job-name=hippo-exploration-noaug
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
set -Eeuo pipefail
campaign="${1:?Pass the frozen campaign directory}"
export PYTHONPATH="${campaign}/source:${campaign}/source/scripts:${campaign}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-cross-mpl-${SLURM_JOB_ID:?}"
cd "$campaign/source"
python_runtime="/home/3160552/.conda/envs/hippocampus/bin/python"
"$python_runtime" -m scripts.cross_dataset_campaign.controller run --stage "$campaign" &
worker_pid=$!
trap 'kill -USR1 "$worker_pid" 2>/dev/null || true' USR1 TERM
set +e
wait "$worker_pid"
exit_code=$?
while kill -0 "$worker_pid" 2>/dev/null; do
    wait "$worker_pid"
    exit_code=$?
done
exit "$exit_code"
