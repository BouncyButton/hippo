#!/usr/bin/env bash
#SBATCH --job-name=outer-focal1-cal
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/outer-focal1-calibration-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/outer-focal1-calibration-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
SOURCE_REPORT=${PROTOCOL}/bands_focal1_calibration.json
CHECKPOINT="$(jq -r '.checkpoint' "${SOURCE_REPORT}")"
CHECKPOINT_CONFIG="$(jq -r '.checkpoint_config' "${SOURCE_REPORT}")"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
cd "${REPO}"

python thesis/new_constraints/bands/calibrate_weight.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --dataset MSD \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --fold 0 \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-config "${CHECKPOINT_CONFIG}" \
  --max-cases 32 \
  --seed 0 \
  --spatial-size 64 64 64 \
  --band-steps 2 \
  --device cuda \
  --amp \
  --bands-focal-gamma 0 \
  --bands-inner-focal-gamma 0 \
  --bands-outer-focal-gamma 1 \
  --output "${PROTOCOL}/bands_focal_inner0_outer1_calibration.json"
