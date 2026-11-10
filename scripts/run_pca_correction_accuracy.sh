#!/usr/bin/env bash
#SBATCH --job-name=pca-correction-accuracy
#SBATCH --partition=stud
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
set -euo pipefail
run_root="${1:?Supply completed 208-case run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/3160552/.conda/envs/hippocampus/bin/python "${run_root}/audit_correction_accuracy.py" "${run_root}"
