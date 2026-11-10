#!/usr/bin/env bash
#SBATCH --job-name=spectral-check-f0
#SBATCH --partition=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
run_root="${1:?Supply the experiment directory}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/3160552/.conda/envs/hippocampus/bin/python "${run_root}/verify.py" "${run_root}/results"
