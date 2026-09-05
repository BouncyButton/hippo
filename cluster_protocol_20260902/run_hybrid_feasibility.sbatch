#!/usr/bin/env bash
#SBATCH --job-name=hybrid-audit-f0
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="/mnt/beegfsstudents/home/3160552/hippo"
PROTOCOL_ROOT="/mnt/beegfsstudents/home/3160552/onecut_protocol_20260902"
AUDIT_ROOT="${REPO_ROOT}/thesis/Surface-normal ordinal LogLTN"
INPUT_ROOT="${PROTOCOL_ROOT}/evaluation/control_bundles"
OUTPUT_ROOT="${PROTOCOL_ROOT}/hybrid_feasibility_20260902"

cd "${REPO_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus

test -f "${INPUT_ROOT}/manifest.json"
test ! -e "${OUTPUT_ROOT}"

PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/hybrid_pycache" python -m py_compile \
  "${AUDIT_ROOT}/surface_normal_ordinal/hybrid.py" \
  "${AUDIT_ROOT}/run_hybrid_feasibility.py"

PYTHONPATH="${AUDIT_ROOT}" python "${AUDIT_ROOT}/run_hybrid_feasibility.py" \
  --input-dir "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --radius-mm 3.0 \
  --ray-step-mm 0.5 \
  --tolerance-mm 1.0 \
  --margin 0.0 \
  --temperature 1.0 \
  --max-surface-points 4096 \
  --seed 0 \
  --device cuda \
  --target-aux-gradient-ratio 0.10 \
  --high-gradient-cap-ratio 0.50 \
  --repair-update-rms 0.02 0.05 0.10
