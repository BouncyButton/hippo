#!/usr/bin/env bash
#SBATCH --job-name=degree-bands-ab
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=degree-bands-ab-%j.out
#SBATCH --error=degree-bands-ab-%j.err
set -euo pipefail

ROOT="${1:?Pass the frozen experiment directory}"
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
REFERENCE_ROOT=/mnt/beegfsstudents/home/3160552
export PYTHONPATH="${ROOT}/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${ROOT}/source"
exec "${PYTHON}" scripts/run_degree_bands_ab.py \
  --root "${ROOT}" \
  --reference-dice "${REFERENCE_ROOT}/regularization_round1_20260916_01/runs/augmentation_seed0" \
  --reference-bands "${REFERENCE_ROOT}/bands_augmented_20260916_04/runs/bands_augmented_seed0" \
  --calibration-checkpoint "${REFERENCE_ROOT}/bands_augmented_20260916_04/calibration_source/checkpoint_latest.pt" \
  --alpha 2 --allow-calibration-mig-profile-change
