#!/usr/bin/env bash
#SBATCH --job-name=unet48-spatial
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-unet48-spatial-${SLURM_JOB_ID}"
/home/3160552/.conda/envs/hippocampus/bin/python "${stage}/diagnose_spatial_errors.py" \
  --pilot-root /mnt/beegfsstudents/home/3160552/compute_budget_unet48_extension_683831 \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --output "${stage}/diagnostic-${SLURM_JOB_ID}.json"
