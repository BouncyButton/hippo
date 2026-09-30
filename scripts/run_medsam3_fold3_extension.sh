#!/usr/bin/env bash
#SBATCH --job-name=medsam3-fold3
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
set -euo pipefail
ROOT="${1:?Pass prepared experiment directory}"
DATA="${2:-/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
cd "$ROOT/source"
exec "$ROOT/venv/bin/python" scripts/medsam3_replication_job.py --root "$ROOT" --data "$DATA" \
  --study docs/experiments/medsam3_fold3_extension_20260929/protocols/study.json
