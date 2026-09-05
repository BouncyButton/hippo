#!/usr/bin/env bash
#SBATCH --job-name=band-loc-sweep
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="/mnt/beegfsstudents/home/3160552/hippo"
PROTOCOL_ROOT="/mnt/beegfsstudents/home/3160552/onecut_protocol_20260902"
AUDIT_ROOT="${REPO_ROOT}/thesis/Surface-normal ordinal LogLTN"
CONTROL_RUN="${PROTOCOL_ROOT}/msd_fold0_none_calibration_seed0"
INPUT_ROOT="${PROTOCOL_ROOT}/hybrid_train32_bundles"
OUTPUT_ROOT="${PROTOCOL_ROOT}/band_location_sweep_train32_20260902"

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus

test -f "${CONTROL_RUN}/checkpoint_best.pt"
test ! -e "${INPUT_ROOT}"
test ! -e "${OUTPUT_ROOT}"

PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/band_location_pycache" python -m py_compile \
  "${AUDIT_ROOT}/surface_normal_ordinal/hybrid.py" \
  "${AUDIT_ROOT}/run_hybrid_feasibility.py" \
  "${AUDIT_ROOT}/run_band_location_sweep.py"

python "${AUDIT_ROOT}/export_cases.py" \
  --pkl datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json datasets/Dataset101_MSD/splits_final.json \
  --dataset MSD \
  --fold 0 \
  --split train \
  --checkpoint "${CONTROL_RUN}/checkpoint_best.pt" \
  --output-dir "${INPUT_ROOT}" \
  --labels-dir datasets/Dataset101_MSD/labelsTr \
  --spatial-size 64 64 64 \
  --device cuda \
  --amp \
  --max-cases 32

PYTHONPATH="${AUDIT_ROOT}" python "${AUDIT_ROOT}/run_band_location_sweep.py" \
  --input-dir "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --location-shares 0.05 0.10 0.20 0.30 \
  --radius-mm 3.0 \
  --ray-step-mm 0.5 \
  --tolerance-mm 1.0 \
  --margin 0.0 \
  --temperature 1.0 \
  --max-surface-points 4096 \
  --max-cases 32 \
  --seed 0 \
  --device cuda \
  --target-aux-gradient-ratio 0.10 \
  --high-gradient-cap-ratio 0.50 \
  --repair-update-rms 0.02 0.05 0.10 \
  --max-union-dice-drop 0.0005 \
  --minimum-passing-update-count 2
