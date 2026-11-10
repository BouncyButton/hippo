#!/usr/bin/env bash
#SBATCH --job-name=pca-teacher-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
run_root="${1:?Supply completed experiment directory}"
export PYTHONPATH="${run_root}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pca-teacher-audit-${SLURM_JOB_ID}"
/home/3160552/.conda/envs/hippocampus/bin/python "${run_root}/audit_teacher.py" \
  --checkpoint /home/3160552/spectral_shape_fold0_20260930_01/baseline_seed0_checkpoint_best.pt \
  --pkl /home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits /home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --basis "${run_root}/results/fixed_basis.pt" \
  --protocol "${run_root}/results/protocol.json" \
  --output "${run_root}/results/teacher_audit_208.json"
