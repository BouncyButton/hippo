#!/usr/bin/env bash
#SBATCH --job-name=onecut-cal-f0
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="/mnt/beegfsstudents/home/3160552/hippo"
PROTOCOL_ROOT="/mnt/beegfsstudents/home/3160552/onecut_protocol_20260902"

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus

python thesis/new_constraints/onecut/calibrate_weight.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --dataset MSD \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --fold 0 \
  --checkpoint "${PROTOCOL_ROOT}/msd_fold0_none_calibration_seed0/checkpoint_latest.pt" \
  --output "${PROTOCOL_ROOT}/onecut_calibration_seed0.json" \
  --max-cases 32 \
  --seed 0 \
  --spatial-size 64 64 64 \
  --onecut-spacing 1 1 1 \
  --onecut-radius-mm 3 \
  --onecut-ray-step-mm 0.5 \
  --onecut-tolerance-mm 1 \
  --onecut-margin 0 \
  --onecut-temperature 1 \
  --onecut-max-surface-points 4096 \
  --device cuda \
  --amp
