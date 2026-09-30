#!/usr/bin/env bash
# Submit from the cluster checkout:
#   sbatch scripts/run_nnunet_cluster.sh
#
# Runs the archived MSD fold 0 (208 train / 52 validation) used by SwinUNETR
# for at most 50 epochs. Validation-Dice early stopping is implemented by
# nnUNetTrainer_50epochsEarlyStopping.
#SBATCH --job-name=nnunet-msd-es
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/run_nnunet_cluster.sh [options]

Options:
  --dataset NAME       Dataset folder (default: Dataset101_MSD)
  --fold N             Existing CV fold (default: 0)
  --env-name NAME      Conda environment (default: hippocampus)
  --run-root PATH      Persistent run directory
  --preprocessed-source PATH  Verified DatasetXXX cache directory to reuse
  -h, --help           Show this help
EOF
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

DATASET="Dataset101_MSD"
FOLD="0"
ENV_NAME="hippocampus"
RUN_ROOT=""
PREPROCESSED_SOURCE=""
TRAINER="nnUNetTrainer_50epochsEarlyStopping"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --preprocessed-source) PREPROCESSED_SOURCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_BASE="${NNUNET_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
  RUN_ROOT="${RUN_BASE}/nnunet_${DATASET}_fold${FOLD}_es50_${SLURM_JOB_ID:-local}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not on PATH." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Process-level parallelism is bounded below. Avoid each worker also spawning
# a full CPU allocation of OpenMP/BLAS/ITK threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export nnUNet_n_proc_DA=2
export nnUNet_def_n_proc=2
export nnUNet_compile="false"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] dataset=${DATASET}, existing_fold=${FOLD}, trainer=${TRAINER}"

python - <<'PY'
import sys
import torch

print(f"[INFO] python={sys.version.split()[0]}, torch={torch.__version__}, cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA device is visible. Submit with --gres=gpu:1.")
print(f"[INFO] gpu_name={torch.cuda.get_device_name(0)}")
PY

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "[INFO] Installing the vendored nnU-Net package into ${ENV_NAME}"
  python -m pip install -e "${REPO_ROOT}/baselines/nnUNet"
fi

DATASET_ID="$(echo "${DATASET}" | sed -E 's/^Dataset([0-9]+)_.+$/\1/')"
SOURCE_DATASET_DIR="${REPO_ROOT}/datasets/${DATASET}"
SOURCE_SPLITS="${SOURCE_DATASET_DIR}/splits_final.json"
RAW_ROOT="${RUN_ROOT}/datasets"
PREPROC_ROOT="${RUN_ROOT}/preprocessed"
RESULTS_ROOT="${RUN_ROOT}/results"
RAW_DATASET_DIR="${RAW_ROOT}/${DATASET}"

if [[ ! -f "${SOURCE_SPLITS}" ]]; then
  echo "[ERROR] Existing split file not found: ${SOURCE_SPLITS}" >&2
  exit 1
fi
python - "${SOURCE_SPLITS}" "${FOLD}" <<'PY'
import json
import sys

splits = json.load(open(sys.argv[1]))
fold = int(sys.argv[2])
split = splits[fold]
if len(split["train"]) != 208 or len(split["val"]) != 52:
    raise RuntimeError(
        f"Expected archived 208/52 split for fold {fold}, got "
        f"{len(split['train'])}/{len(split['val'])}"
    )
if set(split["train"]) & set(split["val"]):
    raise RuntimeError("Train/validation overlap in archived split")
print(f"[INFO] verified archived split fold {fold}: 208 train / 52 validation")
PY

mkdir -p "${RAW_ROOT}" "${PREPROC_ROOT}" "${RESULTS_ROOT}"
if [[ -e "${RAW_DATASET_DIR}" ]]; then
  echo "[ERROR] Fresh run directory expected, but dataset staging path exists: ${RAW_DATASET_DIR}" >&2
  exit 1
fi
mkdir -p "${RAW_DATASET_DIR}"
cp -a "${SOURCE_DATASET_DIR}/." "${RAW_DATASET_DIR}/"
find "${RAW_DATASET_DIR}" -type f -name '._*' -delete

export nnUNet_raw="${RAW_ROOT}"
export nnUNet_preprocessed="${PREPROC_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"

# Only this configuration is trained. The CLI defaults also preprocess 2D
# with eight workers, which is unnecessary for this six-CPU 3D-only job.
if [[ -n "${PREPROCESSED_SOURCE}" ]]; then
  echo "[STAGE] verifying reusable preprocessing cache"
  python "${REPO_ROOT}/scripts/verify_nnunet_cache.py" \
    --cache "${PREPROCESSED_SOURCE}" --raw "${RAW_DATASET_DIR}" \
    --splits "${SOURCE_SPLITS}" --fold "${FOLD}" --output "${RUN_ROOT}/cache_source_verification.json"
  cp -a "${PREPROCESSED_SOURCE}" "${PREPROC_ROOT}/${DATASET}"
else
  echo "[STAGE] preprocessing 3d_fullres"
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity \
    -c 3d_fullres -npfp 2 -np 2 --verbose
fi
cp "${SOURCE_SPLITS}" "${PREPROC_ROOT}/${DATASET}/splits_final.json"
# Small read-only inputs live on node-local disk; checkpoints/results remain
# persistent. No automatic deletion, so an interrupted job preserves evidence.
LOCAL_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/nnunet-${SLURM_JOB_ID:-local}-XXXXXX")"
cp -a "${PREPROC_ROOT}/${DATASET}" "${LOCAL_CACHE}/${DATASET}"
python "${REPO_ROOT}/scripts/verify_nnunet_cache.py" \
  --cache "${LOCAL_CACHE}/${DATASET}" --raw "${RAW_DATASET_DIR}" \
  --splits "${SOURCE_SPLITS}" --fold "${FOLD}" --output "${RUN_ROOT}/cache_training_verification.json"
export nnUNet_preprocessed="${LOCAL_CACHE}"
echo "[STAGE] training starts: $(date -Is), cache=${LOCAL_CACHE}"
nnUNetv2_train "${DATASET_ID}" 3d_fullres "${FOLD}" -tr "${TRAINER}"

MODEL_DIR="${RESULTS_ROOT}/${DATASET}/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}"

# Preserve the validation produced from the stopping checkpoint, then evaluate
# checkpoint_best.pth on the same validation cases. This makes any degradation
# between the best epoch and the stopping epoch directly visible.
if [[ -f "${MODEL_DIR}/validation/summary.json" ]]; then
  cp "${MODEL_DIR}/validation/summary.json" "${MODEL_DIR}/validation_final_summary.json"
fi
nnUNetv2_train "${DATASET_ID}" 3d_fullres "${FOLD}" -tr "${TRAINER}" --val --val_best
if [[ -f "${MODEL_DIR}/validation/summary.json" ]]; then
  cp "${MODEL_DIR}/validation/summary.json" "${MODEL_DIR}/validation_best_summary.json"
fi

python "${REPO_ROOT}/evaluation/summarize_nnunet_training.py" \
  --model-dir "${MODEL_DIR}" \
  --output "${RUN_ROOT}/training_summary.json"

echo "[DONE] run_root=${RUN_ROOT}"
echo "[DONE] summary=${RUN_ROOT}/training_summary.json"
