#!/usr/bin/env bash
#SBATCH --job-name=focal1-tel-eval
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-eval-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/focal1-telemetry-eval-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_telemetry_30epoch_seed0_20260831
REFERENCE_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_30epoch_seed0_20260830
BCE_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_final_seed0_20260807_104915
BCE_ANALYSIS=${BCE_DIR}/fold0_voxel_error_analysis_best

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
cd "${REPO}"

test -s "${RUN_DIR}/completion_manifest.json"
test -s "${RUN_DIR}/telemetry_epoch.csv"
test -s "${RUN_DIR}/telemetry_gradients.csv"

python evaluation/analyze_training_telemetry.py \
  --run-dir "${RUN_DIR}" \
  --reference-metrics "${REFERENCE_DIR}/metrics.csv" \
  --output-dir "${PROTOCOL}/focal1_telemetry_training_analysis"

for CHECKPOINT in checkpoint_latest.pt checkpoint_best.pt; do
  TAG="${CHECKPOINT#checkpoint_}"
  TAG="${TAG%.pt}"
  python evaluation/fold0_voxel_errors.py \
    --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
    --splits-json datasets/Dataset101_MSD/splits_final.json \
    --checkpoint "${RUN_DIR}/${CHECKPOINT}" \
    --output-dir "${RUN_DIR}/fold0_voxel_error_analysis_${TAG}" \
    --dataset MSD \
    --fold 0 \
    --batch-size 2 \
    --num-workers 4 \
    --spatial-size 64 64 64 \
    --device cuda
done

python evaluation/compare_fold0_voxel_errors.py \
  --baseline-dir "${REFERENCE_DIR}/fold0_voxel_error_analysis_best" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_best" \
  --output-dir "${PROTOCOL}/focal1_telemetry_best_vs_prior_focal1_best" \
  --baseline-name "Prior inner focal-1 / outer BCE best" \
  --candidate-name "Telemetry inner focal-1 / outer BCE best" \
  --bootstrap-samples 20000 \
  --seed 20260831

python evaluation/compare_fold0_voxel_errors.py \
  --baseline-dir "${REFERENCE_DIR}/fold0_voxel_error_analysis_latest" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_latest" \
  --output-dir "${PROTOCOL}/focal1_telemetry_epoch30_vs_prior_focal1_epoch30" \
  --baseline-name "Prior inner focal-1 / outer BCE epoch 30" \
  --candidate-name "Telemetry inner focal-1 / outer BCE epoch 30" \
  --bootstrap-samples 20000 \
  --seed 20260831

python evaluation/compare_fold0_voxel_errors.py \
  --baseline-dir "${BCE_ANALYSIS}" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_best" \
  --output-dir "${PROTOCOL}/focal1_telemetry_best_vs_bce_best" \
  --baseline-name "BCE band best" \
  --candidate-name "Telemetry inner focal-1 / outer BCE best" \
  --bootstrap-samples 20000 \
  --seed 20260831

echo "[COMPLETE] telemetry analysis and exact best/latest voxel comparisons"
