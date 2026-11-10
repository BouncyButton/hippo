#!/usr/bin/env bash
#SBATCH --job-name=hippo-band-winner
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=23:50:00
set -Eeuo pipefail
run_root="${1:?Supply frozen campaign root}"
export PYTHONPATH="${run_root}/source"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-band-winner-${SLURM_JOB_ID}"
cd "${run_root}/source"
exec /home/3160552/.conda/envs/hippocampus/bin/python -u -m scripts.band_focus.winner_resume worker --root "${run_root}"
