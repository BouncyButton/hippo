#!/usr/bin/env bash
# Equivariance experiment:
#   sbatch thesis/new_constraints/run_new_constraints_cluster.sh --constraint-set equivariance
# Boundary-band experiment (use the calibrated weight):
#   sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
#     --constraint-set bands --bands-weight 0.04
# Fine-tune the existing fold-0 baseline:
#   sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
#     --constraint-set equivariance --init-checkpoint /absolute/path/to/model.pt
#SBATCH --job-name=swin-new-constraints
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
  sbatch thesis/new_constraints/run_new_constraints_cluster.sh [options]

Options:
  --constraint-set SET          none, equivariance, or bands
                                 (default: equivariance; translation remains an alias)
  --equivariance-weight FLOAT   Override the selected preset
  --translation-size N          Integer translation in voxels (default: 2)
  --equivariance-max-samples N  Extra translated samples per batch; 0 means all
                                 (default: 1)
  --bands-weight FLOAT          Calibrated positive weight required for bands
  --band-steps N                Fixed at 2 for the canonical bands experiment
  --constraint-warmup-epochs N  Linear constraint-weight warmup (default: 5)
  --constraint-eval-every N     Validation constraint-metric interval; 0 means final only
                                (default: 5)

  --dataset MSD|MNI|ADNI|COBRA  Dataset identifier (default: MSD)
  --pkl PATH                    Gzipped dataframe (default: Dataset101_MSD)
  --splits-json PATH            Exact CV split JSON (default: Dataset101_MSD)
  --fold N                      Zero-based fold (default: 0)
  --epochs N                    Training epochs (default: 50)
  --batch-size N                Batch size (default: 1; equivariance adds a forward)
  --num-workers N               Data-loader workers (default: 0)
  --spatial-size "X Y Z"        Network input size (default: "64 64 64")
  --resize                      Resize rather than center crop/pad
  --optim-mode MODE             adamw_0.01 or nnunetv2 (default: adamw_0.01)
  --learning-rate FLOAT         Learning rate (default: 1e-4)
  --weight-decay FLOAT          Weight decay (default: 1e-5)
  --step-size N                 StepLR period (default: 20)
  --adamw-gamma FLOAT           StepLR factor (default: 0.5)
  --seed N                      Random seed (default: 0)
  --no-amp                      Disable CUDA mixed precision

  --output-dir PATH             Exact run directory; must not already exist
  --output-root PATH            Parent for generated run directories
  --resume                      Resume --output-dir/checkpoint_latest.pt
  --init-checkpoint PATH        Initialize a new run from a baseline model
  --env-name NAME               Conda environment (default: hippocampus)
  --wandb                       Enable Weights & Biases
  --wandb-project NAME          W&B project (default: hippopotamus-project)
  --wandb-entity NAME           W&B entity (default: focacciafilippo-bocconi-university)
  -h, --help                    Show this help

The default trains with translation equivariance. Use --constraint-set none
for a control run through the same code path.
EOF
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/thesis/new_constraints/train_swinunetr_constraints.py" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "${PWD}/thesis/new_constraints/train_swinunetr_constraints.py" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

DATASET="MSD"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS_JSON="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"
FOLD="0"
EPOCHS="50"
BATCH_SIZE="1"
NUM_WORKERS="0"
SPATIAL_SIZE="64 64 64"
RESIZE="0"
OPTIM_MODE="adamw_0.01"
LEARNING_RATE="1e-4"
WEIGHT_DECAY="1e-5"
STEP_SIZE="20"
ADAMW_GAMMA="0.5"
SEED="0"
USE_AMP="1"

CONSTRAINT_SET="equivariance"
EQUIVARIANCE_WEIGHT=""
TRANSLATION_SIZE="2"
EQUIVARIANCE_MAX_SAMPLES="1"
BANDS_WEIGHT=""
BAND_STEPS="2"
CONSTRAINT_WARMUP_EPOCHS="5"
CONSTRAINT_EVAL_EVERY="5"

OUTPUT_ROOT="${REPO_ROOT}/models/swin_unetr_new_constraints"
OUTPUT_DIR=""
RESUME="0"
INIT_CHECKPOINT=""
ENV_NAME="hippocampus"
USE_WANDB="0"
WANDB_PROJECT="hippopotamus-project"
WANDB_ENTITY="focacciafilippo-bocconi-university"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --constraint-set) CONSTRAINT_SET="$2"; shift 2 ;;
    --equivariance-weight) EQUIVARIANCE_WEIGHT="$2"; shift 2 ;;
    --translation-size) TRANSLATION_SIZE="$2"; shift 2 ;;
    --equivariance-max-samples) EQUIVARIANCE_MAX_SAMPLES="$2"; shift 2 ;;
    --bands-weight) BANDS_WEIGHT="$2"; shift 2 ;;
    --band-steps) BAND_STEPS="$2"; shift 2 ;;
    --constraint-warmup-epochs) CONSTRAINT_WARMUP_EPOCHS="$2"; shift 2 ;;
    --constraint-eval-every) CONSTRAINT_EVAL_EVERY="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --pkl) PKL="$2"; shift 2 ;;
    --splits-json) SPLITS_JSON="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --spatial-size) SPATIAL_SIZE="$2"; shift 2 ;;
    --resize) RESIZE="1"; shift ;;
    --optim-mode) OPTIM_MODE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --step-size) STEP_SIZE="$2"; shift 2 ;;
    --adamw-gamma) ADAMW_GAMMA="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no-amp) USE_AMP="0"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --wandb) USE_WANDB="1"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "${DATASET}" =~ ^(MSD|MNI|ADNI|COBRA)$ ]]; then
  echo "[ERROR] Invalid --dataset: ${DATASET}" >&2
  exit 2
fi
if [[ ! "${CONSTRAINT_SET}" =~ ^(none|equivariance|bands|translation)$ ]]; then
  echo "[ERROR] Invalid --constraint-set: ${CONSTRAINT_SET}" >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "bands" && -z "${BANDS_WEIGHT}" ]]; then
  echo "[ERROR] --constraint-set bands requires --bands-weight." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "bands" && "${BAND_STEPS}" != "2" ]]; then
  echo "[ERROR] The canonical bands experiment requires --band-steps 2." >&2
  exit 2
fi
if [[ "${RESUME}" == "1" && -n "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] --resume and --init-checkpoint are mutually exclusive." >&2
  exit 2
fi
if [[ ! -f "${PKL}" ]]; then
  echo "[ERROR] Dataset pickle not found: ${PKL}" >&2
  exit 2
fi
if [[ ! -f "${SPLITS_JSON}" ]]; then
  echo "[ERROR] Split JSON not found: ${SPLITS_JSON}" >&2
  exit 2
fi
if [[ -n "${INIT_CHECKPOINT}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] Initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${RESUME}" == "1" && -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] --resume requires the original --output-dir." >&2
  exit 2
fi

read -r -a SPATIAL_SIZE_ARRAY <<< "${SPATIAL_SIZE}"
if [[ ${#SPATIAL_SIZE_ARRAY[@]} -ne 3 ]]; then
  echo "[ERROR] --spatial-size must contain exactly three integers." >&2
  exit 2
fi
for dimension in "${SPATIAL_SIZE_ARRAY[@]}"; do
  if (( dimension <= 0 || dimension % 32 != 0 )); then
    echo "[ERROR] Spatial dimensions must be positive multiples of 32." >&2
    exit 2
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not on PATH." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  RUN_ID="${DATASET,,}_fold${FOLD}_${CONSTRAINT_SET}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] dataset=${DATASET}, fold=${FOLD}, constraints=${CONSTRAINT_SET}, output=${OUTPUT_DIR}"
python - <<'PY'
import monai
import torch

print(f"[INFO] torch={torch.__version__}, monai={monai.__version__}, cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device is visible. Submit with --gres=gpu:1.")
print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
PY

COMMAND=(
  python "${REPO_ROOT}/thesis/new_constraints/train_swinunetr_constraints.py"
  --pkl "${PKL}"
  --dataset "${DATASET}"
  --splits-json "${SPLITS_JSON}"
  --fold "${FOLD}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --spatial-size "${SPATIAL_SIZE_ARRAY[@]}"
  --optim-mode "${OPTIM_MODE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --step-size "${STEP_SIZE}"
  --adamw-gamma "${ADAMW_GAMMA}"
  --seed "${SEED}"
  --constraint-set "${CONSTRAINT_SET}"
  --translation-size "${TRANSLATION_SIZE}"
  --equivariance-max-samples "${EQUIVARIANCE_MAX_SAMPLES}"
  --band-steps "${BAND_STEPS}"
  --constraint-warmup-epochs "${CONSTRAINT_WARMUP_EPOCHS}"
  --constraint-eval-every "${CONSTRAINT_EVAL_EVERY}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)
if [[ -n "${EQUIVARIANCE_WEIGHT}" ]]; then
  COMMAND+=(--equivariance-weight "${EQUIVARIANCE_WEIGHT}")
fi
if [[ -n "${BANDS_WEIGHT}" ]]; then
  COMMAND+=(--bands-weight "${BANDS_WEIGHT}")
fi
if [[ "${RESIZE}" == "1" ]]; then
  COMMAND+=(--resize)
fi
if [[ "${USE_AMP}" == "1" ]]; then
  COMMAND+=(--amp)
else
  COMMAND+=(--no-amp)
fi
if [[ "${RESUME}" == "1" ]]; then
  COMMAND+=(--resume)
fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  COMMAND+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  COMMAND+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
  )
fi

echo "[INFO] Running: ${COMMAND[*]}"
"${COMMAND[@]}"
echo "[DONE] Results written to ${OUTPUT_DIR}"
