#!/usr/bin/env bash
#SBATCH --job-name=medsam3-zero-shot
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="${1:?Pass the prepared experiment directory}"
DATA="${2:-/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD}"
PYTHON="${ROOT}/venv/bin/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/source"
# One process handles all three variants; the gated base cache is node-local.
ARGS=(--root "$ROOT" --data "$DATA")
if [[ "${3:-}" == broadcast ]]; then
  ARGS+=(--broadcast-base)
elif [[ $# -ge 3 ]]; then
  ARGS+=(--base-weights "$3")
fi
exec "$PYTHON" scripts/medsam3_cluster_job.py "${ARGS[@]}"
