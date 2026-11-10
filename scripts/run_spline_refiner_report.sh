#!/usr/bin/env bash
#SBATCH --job-name=spline-refiner-check
#SBATCH --partition=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
run_root="${1:?Supply the experiment output root}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-spline-report-${SLURM_JOB_ID}"
/home/3160552/.conda/envs/hippocampus/bin/python "${run_root}/report.py" "${run_root}/results"
