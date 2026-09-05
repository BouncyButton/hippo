#!/usr/bin/env bash
#SBATCH --job-name=inner-focal05-30ep
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/inner-focal05-30epoch-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/inner-focal05-30epoch-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
CALIBRATION=${PROTOCOL}/bands_focal_inner05_outer0_calibration.json
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner05_outer0_30epoch_seed0_20260830
WEIGHT="$(jq -r '.recommended_bands_weight' "${CALIBRATION}")"

cd "${REPO}"
bash thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight "${WEIGHT}" \
  --bands-calibration-json "${CALIBRATION}" \
  --bands-focal-gamma 0 \
  --bands-inner-focal-gamma 0.5 \
  --bands-outer-focal-gamma 0 \
  --band-steps 2 \
  --epochs 30 \
  --batch-size 1 \
  --num-workers 0 \
  --constraint-warmup-epochs 5 \
  --constraint-eval-every 5 \
  --seed 0 \
  --output-dir "${RUN_DIR}"
