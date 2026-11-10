#!/usr/bin/env bash
#SBATCH --job-name=hippo-adni-baselines
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/adni_baselines_5pct_fold0_seed0_20261013_results"
export PYTHONPATH="${stage}/source:${stage}/source/scripts:${stage}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-adni-mpl-${SLURM_JOB_ID:?}"
/home/3160552/.conda/envs/hippocampus/bin/python -m scripts.adni_baselines.worker \
  --stage "${stage}" --root "${root}"
