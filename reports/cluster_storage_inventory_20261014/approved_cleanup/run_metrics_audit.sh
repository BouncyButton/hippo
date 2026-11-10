#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export nnUNet_compile=false
export MPLCONFIGDIR="/tmp/hippo-storage-audit-${SLURM_JOB_ID}"
stage=/mnt/beegfsstudents/home/3160552/storage_retirement_audit_20261014
for experiment in capped_depth_delay_31976_687658 budget_baselines_94603_684354 budget_bands_94603_684431; do
 /home/3160552/.conda/envs/hippocampus/bin/python "$stage/audit_missing_metrics.py" \
  --root "/mnt/beegfsstudents/home/3160552/$experiment" \
  --output "$stage/results" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
done
