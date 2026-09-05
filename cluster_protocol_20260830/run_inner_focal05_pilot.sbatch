#!/usr/bin/env bash
#SBATCH --job-name=inner-focal05-pilot
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/inner-focal05-pilot-%j.out
#SBATCH --error=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829/inner-focal05-pilot-%j.err
set -euo pipefail

REPO=/mnt/beegfsstudents/home/3160552/hippo
PROTOCOL=/mnt/beegfsstudents/home/3160552/focal1_protocol_20260829
CALIBRATION=${PROTOCOL}/bands_focal_inner05_outer0_calibration.json
RUN_DIR=${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_focal_inner05_outer0_pilot_seed0_20260830
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
  --epochs 5 \
  --batch-size 1 \
  --num-workers 0 \
  --constraint-warmup-epochs 5 \
  --constraint-eval-every 5 \
  --seed 0 \
  --output-dir "${RUN_DIR}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hippocampus
for CHECKPOINT in checkpoint_latest.pt checkpoint_best.pt; do
  TAG="${CHECKPOINT#checkpoint_}"
  TAG="${TAG%.pt}"
  python evaluation/fold0_voxel_errors.py \
    --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
    --splits-json datasets/Dataset101_MSD/splits_final.json \
    --checkpoint "${RUN_DIR}/${CHECKPOINT}" \
    --output-dir "${RUN_DIR}/fold0_voxel_error_analysis_${TAG}" \
    --dataset MSD --fold 0 --batch-size 2 --num-workers 4 \
    --spatial-size 64 64 64 --device cuda
done

python evaluation/gate_focal_pilot.py \
  --run-dir "${RUN_DIR}" \
  --output "${PROTOCOL}/inner_focal05_pilot_gate.json" \
  --variant-name "inner-focal-0.5_outer-bce"

python evaluation/compare_fold0_voxel_errors.py \
  --baseline-dir "${REPO}/models/swin_unetr_new_constraints/msd_fold0_bands_pilot_seed0_20260807_082555/fold0_voxel_error_analysis_best" \
  --candidate-dir "${RUN_DIR}/fold0_voxel_error_analysis_best" \
  --output-dir "${PROTOCOL}/inner_focal05_vs_bce_exact_transitions" \
  --baseline-name "BCE band" \
  --candidate-name "Inner focal-0.5 / outer BCE" \
  --bootstrap-samples 20000 --seed 20260830

SELECTION=${PROTOCOL}/selected_inner_focal05_30epoch.json
python evaluation/select_focal_pilot.py \
  --candidate inner_focal05 "${PROTOCOL}/inner_focal05_pilot_gate.json" "${PROTOCOL}/run_inner_focal05_30epoch.sbatch" \
  --output "${SELECTION}"
LAUNCHER="$(jq -r '.selected.launcher // empty' "${SELECTION}")"
if [[ -n "${LAUNCHER}" ]]; then
  LONG_JOB_ID="$(sbatch --parsable "${LAUNCHER}")"
  echo "[SELECTED] submitted promising 30-epoch inner-focal-0.5 job ${LONG_JOB_ID} from ${LAUNCHER}"
else
  echo "[SELECTED] inner-focal-0.5 did not pass; no 30-epoch job submitted"
fi
