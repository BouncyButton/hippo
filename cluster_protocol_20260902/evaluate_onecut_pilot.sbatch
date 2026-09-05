#!/usr/bin/env bash
#SBATCH --job-name=onecut-eval-f0
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

python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split val \
  --checkpoint "${PROTOCOL_ROOT}/msd_fold0_none_calibration_seed0/checkpoint_best.pt" \
  --output-dir "${PROTOCOL_ROOT}/evaluation/control_bundles" \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --device cuda \
  --amp

python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split val \
  --checkpoint "${PROTOCOL_ROOT}/msd_fold0_onecut_pilot_seed0/checkpoint_best.pt" \
  --output-dir "${PROTOCOL_ROOT}/evaluation/onecut_bundles" \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --device cuda \
  --amp

python "${PROTOCOL_ROOT}/compare_checkpoints.py" \
  --control-bundles "${PROTOCOL_ROOT}/evaluation/control_bundles" \
  --onecut-bundles "${PROTOCOL_ROOT}/evaluation/onecut_bundles" \
  --control-run "${PROTOCOL_ROOT}/msd_fold0_none_calibration_seed0" \
  --onecut-run "${PROTOCOL_ROOT}/msd_fold0_onecut_pilot_seed0" \
  --output-dir "${PROTOCOL_ROOT}/evaluation/comparison"
