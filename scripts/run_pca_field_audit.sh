#!/usr/bin/env bash
#SBATCH --job-name=pca-saved-fields
#SBATCH --partition=stud
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
run_root="${1:?Supply completed 208-case run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/3160552/.conda/envs/hippocampus/bin/python "${run_root}/audit_saved_fields.py" "${run_root}"
