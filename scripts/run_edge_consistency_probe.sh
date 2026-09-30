#!/usr/bin/env bash
#SBATCH --job-name=edge-consistency-probe
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
set -euo pipefail
ROOT="${1:?Pass experiment root}"
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
BASE=/mnt/beegfsstudents/home/3160552
cd "${ROOT}/source"
exec /home/3160552/.conda/envs/hippocampus/bin/python scripts/probe_edge_consistency.py \
  --root "${ROOT}/probe" \
  --reference-bands "${BASE}/bands_augmented_20260916_04/runs/bands_augmented_seed0" \
  --checkpoint "${BASE}/bands_augmented_20260916_04/calibration_source/checkpoint_latest.pt" \
  --case-report "${BASE}/degree_bands_ab_20260924_02_3g/calibration_A.json"
