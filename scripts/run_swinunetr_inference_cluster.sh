#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/run_swinunetr_inference_cluster.sh --checkpoint /path/to/model.pt
#SBATCH --job-name=swinunetr-infer
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/run_swinunetr_inference_cluster.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH              Required path to a SwinUNETR model.pt checkpoint
  --dataset MSD|MNI|ADNI|COBRA   Dataset identifier (default: MSD)
  --pkl PATH                     Dataset pickle (default: Dataset101_MSD pickle)
  --splits-json PATH             CV split file (default: Dataset101_MSD/splits_final.json)
  --fold N                       Fold whose held-out split is inferred (default: 0)
  --split train|val              Cases to infer (default: val)
  --batch-size N                 Inference batch size (default: 2)
  --spatial-size "X Y Z"         Must match training (default: "64 64 64")
  --resize                       Must match training if it was enabled
  --output-dir PATH              Results directory (default: next to checkpoint)
  --env-name NAME                Conda environment (default: hippocampus)
EOF
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DATASET="MSD"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS_JSON="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"
CHECKPOINT=""
FOLD="0"
SPLIT="val"
BATCH_SIZE="2"
SPATIAL_SIZE="64 64 64"
RESIZE="0"
OUTPUT_DIR=""
ENV_NAME="hippocampus"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --pkl) PKL="$2"; shift 2 ;;
    --splits-json) SPLITS_JSON="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --spatial-size) SPATIAL_SIZE="$2"; shift 2 ;;
    --resize) RESIZE="1"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Provide an existing --checkpoint PATH." >&2
  exit 2
fi
if [[ ! -f "${PKL}" || ! -f "${SPLITS_JSON}" ]]; then
  echo "[ERROR] Missing dataset pickle or split file." >&2
  exit 2
fi
read -r -a SPATIAL_SIZE_ARR <<< "${SPATIAL_SIZE}"
if [[ ${#SPATIAL_SIZE_ARR[@]} -ne 3 ]]; then
  echo "[ERROR] --spatial-size must contain three values." >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  CHECKPOINT_DIR="$(dirname "${CHECKPOINT}")"
  OUTPUT_DIR="${CHECKPOINT_DIR}/inference_fold${FOLD}_${SPLIT}_${SLURM_JOB_ID:-local}"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

CMD=(
  python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py"
  --checkpoint "${CHECKPOINT}"
  --pkl "${PKL}"
  --dataset "${DATASET}"
  --splits-json "${SPLITS_JSON}"
  --fold "${FOLD}"
  --split "${SPLIT}"
  --batch-size "${BATCH_SIZE}"
  --spatial-size "${SPATIAL_SIZE_ARR[@]}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${RESIZE}" == "1" ]]; then
  CMD+=(--resize)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
