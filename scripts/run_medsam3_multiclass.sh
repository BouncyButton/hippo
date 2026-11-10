#!/usr/bin/env bash
#SBATCH --job-name=medsam3-ap-f0s46
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
set -euo pipefail
ROOT="${1:?Pass prepared experiment directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
cd "$ROOT/source"
exec "$ROOT/venv/bin/python" scripts/medsam3_multiclass_job.py --root "$ROOT"
