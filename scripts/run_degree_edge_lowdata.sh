#!/usr/bin/env bash
#SBATCH --job-name=degree-edge-lowdata-A
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
ROOT="${1:?Pass experiment root}"
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${ROOT}/source"
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
for SEED in 0 1 2; do
  case "$SEED" in 0) RESERVE=1.2;; 1) RESERVE=0.95;; 2) RESERVE=0.7;; esac
  "$PYTHON" scripts/train_degree_edge_lowdata.py --root "${ROOT}/seed${SEED}" --seed "$SEED" --minimum-free-gib "$RESERVE"
done
"$PYTHON" scripts/summarize_edge_lowdata.py --root "$ROOT" --candidate weighted
