#!/usr/bin/env bash
# Submit from the cluster checkout with, for example:
#   sbatch scripts/run_swinunetr_cluster.sh --fold 0 --epochs 50
#
# Resource directives are intentionally conservative for one 40-GB GPU slice.
# Override them at submission time when needed, e.g.:
#   sbatch --time=12:00:00 --mem=96G scripts/run_swinunetr_cluster.sh
#SBATCH --job-name=swinunetr
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/run_swinunetr_cluster.sh [options]

Options:
  --dataset MSD|MNI|ADNI|COBRA  Dataset identifier used by swin_unetr.py (default: MSD)
  --pkl PATH                    Gzipped dataframe input (default: Dataset101_MSD pickle)
  --splits-json PATH            JSON CV splits (default: Dataset101_MSD/splits_final.json)
  --fold N                      CV fold to train (default: 0)
  --epochs N                    Training epochs (default: 50)
  --batch-size N                Per-GPU batch size (default: 2)
  --spatial-size "X Y Z"        Input ROI size; every dimension must be divisible by 32 (default: "64 64 64")
  --resize                      Resize volumes to --spatial-size instead of centre crop/pad
  --optim-mode adamw_0.01|nnunetv2
                                Optimizer preset from swin_unetr.py (default: adamw_0.01)
  --learning-rate FLOAT         Optimizer learning rate (default: 1e-4)
  --weight-decay FLOAT          Optimizer weight decay (default: 1e-5)
  --step-size N                 AdamW StepLR period (default: 20)
  --adamw-gamma FLOAT           AdamW StepLR decay factor (default: 0.5)
  --env-name NAME               Conda environment (default: hippocampus)
  --model-out-dir PATH          Parent directory for checkpoints (default: models/swin_unetr/<run id>)
  --wandb                       Enable Weights & Biases logging (disabled by default)
  -h, --help                    Show this help

Examples:
  # Recommended first 50-epoch run on the cluster's available MSD dataset.
  sbatch scripts/run_swinunetr_cluster.sh --fold 0 --epochs 50

  # If the 40-GB slice runs out of memory, resubmit with one sample per batch.
  sbatch scripts/run_swinunetr_cluster.sh --batch-size 1 --fold 0
EOF
}

# Slurm copies the batch script to its spool directory before executing it.
# SLURM_SUBMIT_DIR retains the repository directory from which sbatch was run.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DATASET="MSD"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS_JSON="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"
FOLD="0"
EPOCHS="50"
BATCH_SIZE="2"
SPATIAL_SIZE="64 64 64"
RESIZE="0"
OPTIM_MODE="adamw_0.01"
LEARNING_RATE="1e-4"
WEIGHT_DECAY="1e-5"
STEP_SIZE="20"
ADAMW_GAMMA="0.5"
ENV_NAME="hippocampus"
MODEL_OUT_DIR=""
USE_WANDB="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --pkl) PKL="$2"; shift 2 ;;
    --splits-json) SPLITS_JSON="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --spatial-size) SPATIAL_SIZE="$2"; shift 2 ;;
    --resize) RESIZE="1"; shift ;;
    --optim-mode) OPTIM_MODE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --step-size) STEP_SIZE="$2"; shift 2 ;;
    --adamw-gamma) ADAMW_GAMMA="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --model-out-dir) MODEL_OUT_DIR="$2"; shift 2 ;;
    --wandb) USE_WANDB="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! " ${DATASET} " =~ ^\ (MSD|MNI|ADNI|COBRA)\ $ ]]; then
  echo "[ERROR] --dataset must be one of MSD, MNI, ADNI, COBRA." >&2
  exit 2
fi
if [[ ! -f "${PKL}" ]]; then
  echo "[ERROR] Dataset pickle not found: ${PKL}" >&2
  exit 2
fi
if [[ ! -f "${SPLITS_JSON}" ]]; then
  echo "[ERROR] CV split file not found: ${SPLITS_JSON}" >&2
  exit 2
fi

read -r -a SPATIAL_SIZE_ARR <<< "${SPATIAL_SIZE}"
if [[ ${#SPATIAL_SIZE_ARR[@]} -ne 3 ]]; then
  echo "[ERROR] --spatial-size must contain exactly three integers." >&2
  exit 2
fi
for dimension in "${SPATIAL_SIZE_ARR[@]}"; do
  if (( dimension <= 0 || dimension % 32 != 0 )); then
    echo "[ERROR] Each --spatial-size dimension must be positive and divisible by 32: ${SPATIAL_SIZE}" >&2
    exit 2
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not on PATH. Load the cluster miniconda module before submitting." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ -z "${MODEL_OUT_DIR}" ]]; then
  RUN_ID="${DATASET,,}_fold${FOLD}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
  MODEL_OUT_DIR="${REPO_ROOT}/models/swin_unetr/${RUN_ID}"
fi
mkdir -p "${MODEL_OUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] dataset=${DATASET}, fold=${FOLD}, epochs=${EPOCHS}, batch_size=${BATCH_SIZE}"
echo "[INFO] spatial_size=${SPATIAL_SIZE}, optimizer=${OPTIM_MODE}, lr=${LEARNING_RATE}, weight_decay=${WEIGHT_DECAY}, model_out_dir=${MODEL_OUT_DIR}"
python - <<'PY'
import monai
import torch

print(f"[INFO] torch={torch.__version__}, monai={monai.__version__}, cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device is visible. Submit this script with --gres=gpu:1.")
print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
PY

TRAIN_CMD=(
  python "${REPO_ROOT}/baselines/swin_unetr/swin_unetr.py"
  --pkl "${PKL}"
  --dataset "${DATASET}"
  --splits-json "${SPLITS_JSON}"
  --fold "${FOLD}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --spatial-size "${SPATIAL_SIZE_ARR[@]}"
  --optim-mode "${OPTIM_MODE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --model-out-dir "${MODEL_OUT_DIR}"
)
if [[ "${OPTIM_MODE}" == "adamw_0.01" ]]; then
  TRAIN_CMD+=(--step-size "${STEP_SIZE}" --adamw-gamma "${ADAMW_GAMMA}")
fi
if [[ "${RESIZE}" == "1" ]]; then
  TRAIN_CMD+=(--resize)
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  TRAIN_CMD+=(--wandb)
else
  TRAIN_CMD+=(--no-wandb)
fi

echo "[INFO] Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
echo "[DONE] Checkpoint written under ${MODEL_OUT_DIR}"
