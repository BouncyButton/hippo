#!/usr/bin/env bash
#SBATCH --job-name=degree-boundary-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="${1:?Pass the audit directory}"
EXPERIMENT="${2:?Pass the completed A/B experiment directory}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${EXPERIMENT}/source"
cd "${ROOT}"
exec /home/3160552/.conda/envs/hippocampus/bin/python "${ROOT}/audit_degree_boundary.py" \
  --experiment "${EXPERIMENT}" --output "${ROOT}/results"
