#!/usr/bin/env bash
#SBATCH --job-name=hybrid30-opdiag
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/hybrid30-opdiag-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/hybrid30-opdiag-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
BCE_RUN=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_final_seed0_20260807_104915
HYBRID_RUN=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner1_outer0_30epoch_seed0_20260830
OUTPUT=${PROTOCOL}/hybrid30_best_vs_bce_best_operating_points

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
cd "${REPO}"

python evaluation/analyze_focal_operating_points.py \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --baseline-checkpoint "${BCE_RUN}/checkpoint_best.pt" \
  --candidate-checkpoint "${HYBRID_RUN}/checkpoint_best.pt" \
  --baseline-error-dir "${BCE_RUN}/fold0_voxel_error_analysis_best" \
  --candidate-error-dir "${HYBRID_RUN}/fold0_voxel_error_analysis_best" \
  --output-dir "${OUTPUT}" \
  --baseline-name "BCE band best" \
  --candidate-name "Inner focal-1 / outer BCE best" \
  --dataset MSD \
  --fold 0 \
  --batch-size 2 \
  --num-workers 4 \
  --spatial-size 64 64 64 \
  --device cuda \
  --background-bias-min -2 \
  --background-bias-max 8 \
  --background-bias-steps 201 \
  --posterior-bias-min -3 \
  --posterior-bias-max 3 \
  --posterior-bias-steps 121 \
  --coordinate-iterations 3
