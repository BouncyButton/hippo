#!/usr/bin/env bash
#SBATCH --job-name=onecut-v-band
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_ROOT="/mnt/beegfsstudents/home/3160552/hippo"
PROTOCOL_ROOT="/mnt/beegfsstudents/home/3160552/onecut_protocol_20260902"
BAND_RUN="${REPO_ROOT}/models/swin_unetr_new_constraints/msd_fold0_bands_pilot_seed0_20260807_082555"
EVALUATION_ROOT="${PROTOCOL_ROOT}/evaluation/onecut_vs_bands"

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus

test -f "${BAND_RUN}/checkpoint_best.pt"
test -f "${BAND_RUN}/final_metrics.json"
test -f "${PROTOCOL_ROOT}/msd_fold0_onecut_pilot_seed0/final_metrics.json"
test -f "${PROTOCOL_ROOT}/evaluation/onecut_bundles/manifest.json"
test ! -e "${EVALUATION_ROOT}"

mkdir -p "${EVALUATION_ROOT}/band_bundles"

python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split val \
  --checkpoint "${BAND_RUN}/checkpoint_best.pt" \
  --output-dir "${EVALUATION_ROOT}/band_bundles" \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --device cuda \
  --amp

python "${PROTOCOL_ROOT}/compare_checkpoints.py" \
  --control-bundles "${EVALUATION_ROOT}/band_bundles" \
  --onecut-bundles "${PROTOCOL_ROOT}/evaluation/onecut_bundles" \
  --control-run "${BAND_RUN}" \
  --onecut-run "${PROTOCOL_ROOT}/msd_fold0_onecut_pilot_seed0" \
  --output-dir "${EVALUATION_ROOT}/comparison"

