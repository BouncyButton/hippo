#!/usr/bin/env bash
#SBATCH --job-name=hybrid30-eval
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/hybrid30-eval-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/hybrid30-eval-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_30epoch_seed0_20260830
BCE_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_final_seed0_20260807_104915
BCE_ANALYSIS=${BCE_DIR}/fold0_voxel_error_analysis_best

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
cd "${REPO}"

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
  --baseline-dir "${BCE_ANALYSIS}" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_best" \
  --output-dir "${PROTOCOL}/hybrid30_best_vs_bce_best_exact" \
  --baseline-name "BCE band best" \
  --candidate-name "Inner focal-1 / outer BCE best" \
  --bootstrap-samples 20000 \
  --seed 20260830

python evaluation/compare_fold0_voxel_errors.py \
  --baseline-dir "${BCE_ANALYSIS}" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_latest" \
  --output-dir "${PROTOCOL}/hybrid30_epoch30_vs_bce_best_exact" \
  --baseline-name "BCE band best" \
  --candidate-name "Inner focal-1 / outer BCE epoch 30" \
  --bootstrap-samples 20000 \
  --seed 20260830

echo "[COMPLETE] evaluated hybrid best and epoch-30 checkpoints against BCE best"
