#!/usr/bin/env bash
#SBATCH --job-name=edge-generalization-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT="${1:?Pass audit root}"
BASE=/mnt/beegfsstudents/home/3160552
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "${ROOT}/source"
exec /home/3160552/.conda/envs/hippocampus/bin/python scripts/audit_edge_generalization.py \
  --output "${ROOT}/results" \
  --edge-experiment "${BASE}/edge_consistency_training_20260924_01" \
  --reference-bands "${BASE}/bands_augmented_20260916_04/runs/bands_augmented_seed0" \
  --reference-dice "${BASE}/regularization_round1_20260916_01/runs/augmentation_seed0"
