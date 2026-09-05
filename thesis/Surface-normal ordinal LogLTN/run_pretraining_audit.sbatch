#!/usr/bin/env bash
#SBATCH --job-name=ordinal-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=ordinal-audit-%j.out
#SBATCH --error=ordinal-audit-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch "thesis/Surface-normal ordinal LogLTN/run_pretraining_audit.sbatch" \
    --checkpoint /absolute/path/checkpoint_best.pt \
    --output-root /absolute/path/ordinal_audit

Or reuse saved SwinUNETR probabilities:
  sbatch "thesis/Surface-normal ordinal LogLTN/run_pretraining_audit.sbatch" \
    --inference-dir /absolute/path/inference_folder \
    --output-root /absolute/path/ordinal_audit

Options:
  --checkpoint PATH       Full or standalone SwinUNETR checkpoint
  --inference-dir PATH    Folder with prediction_index.csv and predictions/
  --output-root PATH      New parent folder for bundles and results (required)
  --split train|val       Dataset split (default: val)
  --max-cases N           Optional deterministic prefix for a smoke run
  --interfaces MODE       outer or outer+ap (default: outer)
  --env-name NAME         Conda environment (default: hippocampus)
EOF
}

REPO="${SLURM_SUBMIT_DIR:-/mnt/beegfsstudents/home/3160552/hippo}"
PKL="${REPO}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS="${REPO}/datasets/Dataset101_MSD/splits_final.json"
LABELS_DIR="${REPO}/datasets/Dataset101_MSD/labelsTr"
CHECKPOINT=""
INFERENCE_DIR=""
OUTPUT_ROOT=""
SPLIT="val"
MAX_CASES=""
INTERFACES="outer"
ENV_NAME="hippocampus"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --inference-dir) INFERENCE_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    --interfaces) INTERFACES="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "--output-root is required" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT}" && -n "${INFERENCE_DIR}" ]] || [[ -z "${CHECKPOINT}" && -z "${INFERENCE_DIR}" ]]; then
  echo "Pass exactly one of --checkpoint and --inference-dir" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output root already exists; choose a new path: ${OUTPUT_ROOT}" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO}"

mkdir -p "${OUTPUT_ROOT}/bundles"
SOURCE_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  SOURCE_ARGS+=(--checkpoint "${CHECKPOINT}" --device cuda)
else
  SOURCE_ARGS+=(--inference-dir "${INFERENCE_DIR}" --device cpu)
fi
LIMIT_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  LIMIT_ARGS+=(--max-cases "${MAX_CASES}")
fi

python "thesis/Surface-normal ordinal LogLTN/export_cases.py" \
  --pkl "${PKL}" \
  --splits-json "${SPLITS}" \
  --dataset MSD \
  --fold 0 \
  --split "${SPLIT}" \
  --labels-dir "${LABELS_DIR}" \
  --spatial-size 64 64 64 \
  --output-dir "${OUTPUT_ROOT}/bundles" \
  "${SOURCE_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

python "thesis/Surface-normal ordinal LogLTN/run_audit.py" \
  --input-dir "${OUTPUT_ROOT}/bundles" \
  --output-dir "${OUTPUT_ROOT}/results" \
  --interfaces "${INTERFACES}" \
  --radius-mm 3 \
  --ray-step-mm 0.5 \
  --delta-mm 1 \
  --localization-tolerance-mm 1 \
  --repair-update-rms 0.10 \
  --target-aux-gradient-ratio 0.10 \
  --device cuda

echo "[COMPLETE] ${OUTPUT_ROOT}/results/REPORT.md"
