#!/usr/bin/env bash
#SBATCH --job-name=focal1-telemetry-30ep
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-30epoch-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-30epoch-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
CALIBRATION=${PROTOCOL}/bands_focal_inner1_outer0_telemetry_v2_calibration.json
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_telemetry_30epoch_seed0_20260831
WEIGHT="$(jq -r '.recommended_bands_weight' "${CALIBRATION}")"

cd "${REPO}"
bash thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight "${WEIGHT}" \
  --bands-calibration-json "${CALIBRATION}" \
  --bands-focal-gamma 0 \
  --bands-inner-focal-gamma 1 \
  --bands-outer-focal-gamma 0 \
  --band-steps 2 \
  --epochs 30 \
  --batch-size 1 \
  --num-workers 0 \
  --constraint-warmup-epochs 5 \
  --constraint-eval-every 5 \
  --seed 0 \
  --telemetry \
  --telemetry-probe-epochs "1 3 5 8 10 12 14 16 18 20 25 30" \
  --telemetry-probe-cases 2 \
  --telemetry-spatial-cases 2 \
  --output-dir "${RUN_DIR}"
