#!/usr/bin/env bash
#SBATCH --job-name=voxel-strata
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/voxel-strata-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/voxel-strata-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
BCE_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_final_seed0_20260807_104915
FOCAL_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_telemetry_30epoch_seed0_20260831
OUTPUT_TAG=${OUTPUT_TAG:-full_20260901}
MAX_CASES=${MAX_CASES:-0}
OUTPUT_DIR=${PROTOCOL}/offline_voxel_stratification_${OUTPUT_TAG}

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
cd "${REPO}"

EXTRA_ARGS=()
if [[ "${MAX_CASES}" != "0" ]]; then
  EXTRA_ARGS+=(--max-cases "${MAX_CASES}")
fi

python evaluation/stratify_boundary_voxels.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --bce-checkpoint "${BCE_DIR}/checkpoint_best.pt" \
  --focal-best-checkpoint "${FOCAL_DIR}/checkpoint_best.pt" \
  --focal-final-checkpoint "${FOCAL_DIR}/checkpoint_latest.pt" \
  --bce-error-dir "${BCE_DIR}/fold0_voxel_error_analysis_best" \
  --focal-error-dir "${FOCAL_DIR}/fold0_voxel_error_analysis_best" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset MSD \
  --fold 0 \
  --batch-size 2 \
  --num-workers 4 \
  --spatial-size 64 64 64 \
  --band-steps 2 \
  --device cuda \
  "${EXTRA_ARGS[@]}"

echo "[COMPLETE] ${OUTPUT_DIR}"
