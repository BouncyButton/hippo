#!/usr/bin/env bash
#SBATCH --job-name=class-tv-30ep
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/class-tversky-30epoch-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/class-tversky-30epoch-%j.err

set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
CALIBRATION=${PROTOCOL}/bands_class_tversky_fp060_fn040_calibration.json
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_class_tversky_fp060_fn040_30epoch_seed0_20260901
WEIGHT="$(jq -r '.recommended_bands_weight' "${CALIBRATION}")"
cd "${REPO}"

bash thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set bands \
  --bands-weight "${WEIGHT}" \
  --bands-calibration-json "${CALIBRATION}" \
  --bands-loss-type class_tversky \
  --tversky-fp-weight 0.60 \
  --tversky-fn-weight 0.40 \
  --bands-focal-gamma 0 \
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
