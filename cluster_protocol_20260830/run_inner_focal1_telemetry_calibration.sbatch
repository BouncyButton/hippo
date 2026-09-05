#!/usr/bin/env bash
#SBATCH --job-name=focal1-tel-cal
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-calibration-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-calibration-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
SOURCE_REPORT=${PROTOCOL}/bands_focal_inner1_outer0_calibration.json
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
  --bands-focal-gamma 0 \
  --bands-inner-focal-gamma 1 \
  --bands-outer-focal-gamma 0 \
  --device cuda \
  --amp \
  --output "${PROTOCOL}/bands_focal_inner1_outer0_telemetry_v2_calibration.json"
