#!/usr/bin/env bash
#SBATCH --job-name=medsam3-ap-report
#SBATCH --partition=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="${1:?Pass prepared study directory}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
cd "$ROOT/source"
exec "$ROOT/venv/bin/python" scripts/medsam3_multiclass_replication.py aggregate --root "$ROOT"
