#!/usr/bin/env bash
# Equivariance experiment:
#   sbatch thesis/new_constraints/run_new_constraints_cluster.sh --constraint-set equivariance
# Boundary-band experiment (use the calibrated weight):
#   sbatch thesis/new_constraints/run_new_constraints_cluster.sh \
#     --constraint-set bands --bands-weight 0.04 \
#     --bands-calibration-json /absolute/path/to/bands_calibration.json
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
  --constraint-set SET          none, equivariance, bands, onecut, teacher, ap_cut,
                                 ap_plane, ap_plane_location, ap_plane_ce_control,
                                 or perimeter_profile
                                 (default: equivariance; translation remains an alias)
  --equivariance-weight FLOAT   Override the selected preset
  --supervised-loss NAME        dice (default) or dice_ce
  --ce-weight FLOAT             CE coefficient for dice_ce (default: 1)
  --calibration-diagnostics     Log validation probability diagnostics by region
  --teacher-weight FLOAT        Experimental KL weight from training-only audit
  --teacher-views N             Sampled teacher translations (default: 2 of 12)
  --teacher-temperature FLOAT   Teacher/student temperature (default: 1)
  --teacher-support union|common Fixed all-shift overlap for stochastic teacher (default: union)
  --translation-augmentation    Supervised +/-2 axis shifts with probability 1/2
  --ap-cut-weight FLOAT         Experimental structured A/P weight from audit
  --ap-axis N                   Verified spatial tensor axis, required for ap_cut
  --ap-anterior-side low|high   Verified orientation, required for ap_cut
  --ap-temperature FLOAT        Cut-score temperature (default: 1)
  --ap-plane-weight FLOAT       Calibrated A/P-plane weight
  --ap-plane-calibration-json P Completed training-only plane calibration
  --ap-plane-axis N             Stored coronal axis (Task04: 1)
  --ap-plane-anterior-side S    low or high (Task04: high)
  --ap-plane-margin FLOAT       A/P logit margin (default: 0)
  --perimeter-profile-weight F  Weight selected by the training-only gradient audit
  --translation-size N          Integer translation in voxels (default: 2)
  --equivariance-max-samples N  Extra translated samples per batch; 0 means all
                                 (default: 1)
  --bands-weight FLOAT          Calibrated positive weight required for bands
  --bands-calibration-json PATH Completed calibration report required for bands
  --band-steps N                Fixed at 2 for the canonical bands experiment
  --bands-focal-gamma FLOAT     Focal exponent for bands (default: 0, exact BCE)
  --bands-degree-alpha FLOAT    GT face-exposure weighting (default: 0, disabled)
  --bands-degree-normalization inner|surface  Entire inner band (A) or surface only (B)
  --bands-inner-focal-gamma F   Optional inner exponent (defaults to shared gamma)
  --bands-outer-focal-gamma F   Optional outer exponent (defaults to shared gamma)
  --bands-loss-type TYPE        focal_bce or class_tversky (default: focal_bce)
  --tversky-fp-weight FLOAT     Class Tversky FP penalty (default: 0.60)
  --tversky-fn-weight FLOAT     Class Tversky FN penalty (default: 0.40)
  --onecut-weight FLOAT         Calibrated positive weight required for onecut
  --onecut-calibration-json P   Completed onecut calibration report
  --onecut-spacing "D H W"      Physical voxel spacing (default: "1 1 1")
  --onecut-radius-mm FLOAT      Normal-ray radius (default: 3)
  --onecut-ray-step-mm FLOAT    Sampling interval (default: 0.5)
  --onecut-tolerance-mm FLOAT   Allowed cut displacement (default: 1)
  --onecut-margin FLOAT         Unary logit margin (default: 0)
  --onecut-temperature FLOAT    Logical truth temperature (default: 1)
  --onecut-max-points N         Maximum deterministic faces/case (default: 4096)
  --constraint-warmup-epochs N  Linear constraint-weight warmup (default: 5)
  --constraint-eval-every N     Validation constraint-metric interval; 0 means final only
                                (default: 5)
  --telemetry                   Record validation focus and fixed-case gradients
  --telemetry-probe-epochs LIST Space-delimited probe epochs (default: 1 3 5 8 10 12 14 16 18 20 25 30)
  --telemetry-probe-cases N     Fixed validation cases per gradient probe (default: 2)
  --telemetry-spatial-cases N   Cases with saved spatial maps per probe (default: 2)

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
SUPERVISED_LOSS="dice"
CE_WEIGHT="1"
CALIBRATION_DIAGNOSTICS="0"
TEACHER_WEIGHT=""
TEACHER_VIEWS="2"
TEACHER_TEMPERATURE="1"
TEACHER_SUPPORT="union"
TRANSLATION_AUGMENTATION="0"
AP_CUT_WEIGHT=""
AP_AXIS=""
AP_ANTERIOR_SIDE=""
AP_TEMPERATURE="1"
AP_PLANE_WEIGHT=""
AP_PLANE_CALIBRATION_JSON=""
AP_PLANE_AXIS=""
AP_PLANE_ANTERIOR_SIDE=""
AP_PLANE_MARGIN="0"
PERIMETER_PROFILE_WEIGHT=""

CONSTRAINT_SET="equivariance"
EQUIVARIANCE_WEIGHT=""
TRANSLATION_SIZE="2"
EQUIVARIANCE_MAX_SAMPLES="1"
BANDS_WEIGHT=""
BANDS_CALIBRATION_JSON=""
BAND_STEPS="2"
BANDS_FOCAL_GAMMA="0"
BANDS_DEGREE_ALPHA="0"
BANDS_DEGREE_NORMALIZATION="inner"
BANDS_INNER_FOCAL_GAMMA=""
BANDS_OUTER_FOCAL_GAMMA=""
BANDS_LOSS_TYPE="focal_bce"
TVERSKY_FP_WEIGHT="0.60"
TVERSKY_FN_WEIGHT="0.40"
ONECUT_WEIGHT=""
ONECUT_CALIBRATION_JSON=""
ONECUT_SPACING="1 1 1"
ONECUT_RADIUS_MM="3"
ONECUT_RAY_STEP_MM="0.5"
ONECUT_TOLERANCE_MM="1"
ONECUT_MARGIN="0"
ONECUT_TEMPERATURE="1"
ONECUT_MAX_POINTS="4096"
CONSTRAINT_WARMUP_EPOCHS="5"
CONSTRAINT_EVAL_EVERY="5"
USE_TELEMETRY="0"
TELEMETRY_PROBE_EPOCHS="1 3 5 8 10 12 14 16 18 20 25 30"
TELEMETRY_PROBE_CASES="2"
TELEMETRY_SPATIAL_CASES="2"

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
    --supervised-loss) SUPERVISED_LOSS="$2"; shift 2 ;;
    --ce-weight) CE_WEIGHT="$2"; shift 2 ;;
    --calibration-diagnostics) CALIBRATION_DIAGNOSTICS="1"; shift ;;
    --teacher-weight) TEACHER_WEIGHT="$2"; shift 2 ;;
    --teacher-views) TEACHER_VIEWS="$2"; shift 2 ;;
    --teacher-temperature) TEACHER_TEMPERATURE="$2"; shift 2 ;;
    --teacher-support) TEACHER_SUPPORT="$2"; shift 2 ;;
    --translation-augmentation) TRANSLATION_AUGMENTATION="1"; shift ;;
    --ap-cut-weight) AP_CUT_WEIGHT="$2"; shift 2 ;;
    --ap-axis) AP_AXIS="$2"; shift 2 ;;
    --ap-anterior-side) AP_ANTERIOR_SIDE="$2"; shift 2 ;;
    --ap-temperature) AP_TEMPERATURE="$2"; shift 2 ;;
    --ap-plane-weight) AP_PLANE_WEIGHT="$2"; shift 2 ;;
    --ap-plane-calibration-json) AP_PLANE_CALIBRATION_JSON="$2"; shift 2 ;;
    --ap-plane-axis) AP_PLANE_AXIS="$2"; shift 2 ;;
    --ap-plane-anterior-side) AP_PLANE_ANTERIOR_SIDE="$2"; shift 2 ;;
    --ap-plane-margin) AP_PLANE_MARGIN="$2"; shift 2 ;;
    --perimeter-profile-weight) PERIMETER_PROFILE_WEIGHT="$2"; shift 2 ;;
    --equivariance-weight) EQUIVARIANCE_WEIGHT="$2"; shift 2 ;;
    --translation-size) TRANSLATION_SIZE="$2"; shift 2 ;;
    --equivariance-max-samples) EQUIVARIANCE_MAX_SAMPLES="$2"; shift 2 ;;
    --bands-weight) BANDS_WEIGHT="$2"; shift 2 ;;
    --bands-calibration-json) BANDS_CALIBRATION_JSON="$2"; shift 2 ;;
    --band-steps) BAND_STEPS="$2"; shift 2 ;;
    --bands-focal-gamma) BANDS_FOCAL_GAMMA="$2"; shift 2 ;;
    --bands-degree-alpha) BANDS_DEGREE_ALPHA="$2"; shift 2 ;;
    --bands-degree-normalization) BANDS_DEGREE_NORMALIZATION="$2"; shift 2 ;;
    --bands-inner-focal-gamma) BANDS_INNER_FOCAL_GAMMA="$2"; shift 2 ;;
    --bands-outer-focal-gamma) BANDS_OUTER_FOCAL_GAMMA="$2"; shift 2 ;;
    --bands-loss-type) BANDS_LOSS_TYPE="$2"; shift 2 ;;
    --tversky-fp-weight) TVERSKY_FP_WEIGHT="$2"; shift 2 ;;
    --tversky-fn-weight) TVERSKY_FN_WEIGHT="$2"; shift 2 ;;
    --onecut-weight) ONECUT_WEIGHT="$2"; shift 2 ;;
    --onecut-calibration-json) ONECUT_CALIBRATION_JSON="$2"; shift 2 ;;
    --onecut-spacing) ONECUT_SPACING="$2"; shift 2 ;;
    --onecut-radius-mm) ONECUT_RADIUS_MM="$2"; shift 2 ;;
    --onecut-ray-step-mm) ONECUT_RAY_STEP_MM="$2"; shift 2 ;;
    --onecut-tolerance-mm) ONECUT_TOLERANCE_MM="$2"; shift 2 ;;
    --onecut-margin) ONECUT_MARGIN="$2"; shift 2 ;;
    --onecut-temperature) ONECUT_TEMPERATURE="$2"; shift 2 ;;
    --onecut-max-points) ONECUT_MAX_POINTS="$2"; shift 2 ;;
    --constraint-warmup-epochs) CONSTRAINT_WARMUP_EPOCHS="$2"; shift 2 ;;
    --constraint-eval-every) CONSTRAINT_EVAL_EVERY="$2"; shift 2 ;;
    --telemetry) USE_TELEMETRY="1"; shift ;;
    --telemetry-probe-epochs) TELEMETRY_PROBE_EPOCHS="$2"; shift 2 ;;
    --telemetry-probe-cases) TELEMETRY_PROBE_CASES="$2"; shift 2 ;;
    --telemetry-spatial-cases) TELEMETRY_SPATIAL_CASES="$2"; shift 2 ;;
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
if [[ ! "${CONSTRAINT_SET}" =~ ^(none|equivariance|bands|onecut|translation|teacher|ap_cut|ap_plane|ap_plane_location|ap_plane_ce_control|perimeter_profile)$ ]]; then
  echo "[ERROR] Invalid --constraint-set: ${CONSTRAINT_SET}" >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" =~ ^(ap_plane|ap_plane_location|ap_plane_ce_control)$ && ( -z "${AP_PLANE_WEIGHT}" || -z "${AP_PLANE_CALIBRATION_JSON}" || -z "${AP_PLANE_AXIS}" || -z "${AP_PLANE_ANTERIOR_SIDE}" ) ]]; then
  echo "[ERROR] A/P-plane runs require their weight, calibration JSON, axis and anterior side." >&2
  exit 2
fi
if [[ ! "${SUPERVISED_LOSS}" =~ ^(dice|dice_ce)$ ]]; then
  echo "[ERROR] --supervised-loss must be dice or dice_ce." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "teacher" && -z "${TEACHER_WEIGHT}" ]]; then
  echo "[ERROR] teacher requires an explicit --teacher-weight from a training-only audit." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "ap_cut" && ( -z "${AP_CUT_WEIGHT}" || -z "${AP_AXIS}" || -z "${AP_ANTERIOR_SIDE}" ) ]]; then
  echo "[ERROR] ap_cut requires --ap-cut-weight, --ap-axis and --ap-anterior-side." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "perimeter_profile" && -z "${PERIMETER_PROFILE_WEIGHT}" ]]; then
  echo "[ERROR] perimeter_profile requires --perimeter-profile-weight." >&2
  exit 2
fi
if [[ "${SUPERVISED_LOSS}" != "dice" && "${CONSTRAINT_SET}" =~ ^(bands|onecut|ap_plane|ap_plane_location|ap_plane_ce_control)$ ]]; then
  echo "[ERROR] calibrated constraint runs are bound to Dice-only supervision." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "onecut" && -z "${ONECUT_WEIGHT}" ]]; then
  echo "[ERROR] --constraint-set onecut requires --onecut-weight." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "onecut" && -z "${ONECUT_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] --constraint-set onecut requires --onecut-calibration-json." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "bands" && -z "${BANDS_WEIGHT}" ]]; then
  echo "[ERROR] --constraint-set bands requires --bands-weight." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "bands" && -z "${BANDS_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] --constraint-set bands requires --bands-calibration-json." >&2
  exit 2
fi
if [[ "${CONSTRAINT_SET}" == "bands" && "${BAND_STEPS}" != "2" ]]; then
  echo "[ERROR] The canonical bands experiment requires --band-steps 2." >&2
  exit 2
fi
if [[ "${USE_TELEMETRY}" == "1" && "${CONSTRAINT_SET}" != "bands" ]]; then
  echo "[ERROR] --telemetry currently requires --constraint-set bands." >&2
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
if [[ -n "${BANDS_CALIBRATION_JSON}" && ! -f "${BANDS_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] Calibration report not found: ${BANDS_CALIBRATION_JSON}" >&2
  exit 2
fi
if [[ -n "${ONECUT_CALIBRATION_JSON}" && ! -f "${ONECUT_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] One-cut calibration report not found: ${ONECUT_CALIBRATION_JSON}" >&2
  exit 2
fi
if [[ -n "${AP_PLANE_CALIBRATION_JSON}" && ! -f "${AP_PLANE_CALIBRATION_JSON}" ]]; then
  echo "[ERROR] A/P-plane calibration report not found: ${AP_PLANE_CALIBRATION_JSON}" >&2
  exit 2
fi
if [[ "${RESUME}" == "1" && -z "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] --resume requires the original --output-dir." >&2
  exit 2
fi

read -r -a SPATIAL_SIZE_ARRAY <<< "${SPATIAL_SIZE}"
read -r -a ONECUT_SPACING_ARRAY <<< "${ONECUT_SPACING}"
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
if [[ ${#ONECUT_SPACING_ARRAY[@]} -ne 3 ]]; then
  echo "[ERROR] --onecut-spacing must contain exactly three values." >&2
  exit 2
fi

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
  --supervised-loss "${SUPERVISED_LOSS}"
  --ce-weight "${CE_WEIGHT}"
  --teacher-views "${TEACHER_VIEWS}"
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --teacher-support "${TEACHER_SUPPORT}"
  --ap-temperature "${AP_TEMPERATURE}"
  --ap-plane-margin "${AP_PLANE_MARGIN}"
  --translation-size "${TRANSLATION_SIZE}"
  --equivariance-max-samples "${EQUIVARIANCE_MAX_SAMPLES}"
  --band-steps "${BAND_STEPS}"
  --bands-focal-gamma "${BANDS_FOCAL_GAMMA}"
  --bands-degree-alpha "${BANDS_DEGREE_ALPHA}"
  --bands-degree-normalization "${BANDS_DEGREE_NORMALIZATION}"
  --bands-loss-type "${BANDS_LOSS_TYPE}"
  --tversky-fp-weight "${TVERSKY_FP_WEIGHT}"
  --tversky-fn-weight "${TVERSKY_FN_WEIGHT}"
  --onecut-spacing "${ONECUT_SPACING_ARRAY[@]}"
  --onecut-radius-mm "${ONECUT_RADIUS_MM}"
  --onecut-ray-step-mm "${ONECUT_RAY_STEP_MM}"
  --onecut-tolerance-mm "${ONECUT_TOLERANCE_MM}"
  --onecut-margin "${ONECUT_MARGIN}"
  --onecut-temperature "${ONECUT_TEMPERATURE}"
  --onecut-max-surface-points "${ONECUT_MAX_POINTS}"
  --constraint-warmup-epochs "${CONSTRAINT_WARMUP_EPOCHS}"
  --constraint-eval-every "${CONSTRAINT_EVAL_EVERY}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)
if [[ "${CALIBRATION_DIAGNOSTICS}" == "1" ]]; then
  COMMAND+=(--calibration-diagnostics)
fi
if [[ "${TRANSLATION_AUGMENTATION}" == "1" ]]; then
  COMMAND+=(--translation-augmentation)
fi
if [[ -n "${TEACHER_WEIGHT}" ]]; then
  COMMAND+=(--teacher-weight "${TEACHER_WEIGHT}")
fi
if [[ -n "${AP_CUT_WEIGHT}" ]]; then
  COMMAND+=(--ap-cut-weight "${AP_CUT_WEIGHT}")
fi
if [[ -n "${AP_PLANE_WEIGHT}" ]]; then
  COMMAND+=(--ap-plane-weight "${AP_PLANE_WEIGHT}")
fi
if [[ -n "${AP_PLANE_CALIBRATION_JSON}" ]]; then
  COMMAND+=(--ap-plane-calibration-json "${AP_PLANE_CALIBRATION_JSON}")
fi
if [[ -n "${AP_PLANE_AXIS}" ]]; then
  COMMAND+=(--ap-plane-axis "${AP_PLANE_AXIS}")
fi
if [[ -n "${AP_PLANE_ANTERIOR_SIDE}" ]]; then
  COMMAND+=(--ap-plane-anterior-side "${AP_PLANE_ANTERIOR_SIDE}")
fi
if [[ -n "${PERIMETER_PROFILE_WEIGHT}" ]]; then
  COMMAND+=(--perimeter-profile-weight "${PERIMETER_PROFILE_WEIGHT}")
fi
if [[ -n "${AP_AXIS}" ]]; then
  COMMAND+=(--ap-axis "${AP_AXIS}")
fi
if [[ -n "${AP_ANTERIOR_SIDE}" ]]; then
  COMMAND+=(--ap-anterior-side "${AP_ANTERIOR_SIDE}")
fi
if [[ -n "${BANDS_INNER_FOCAL_GAMMA}" ]]; then
  COMMAND+=(--bands-inner-focal-gamma "${BANDS_INNER_FOCAL_GAMMA}")
fi
if [[ -n "${BANDS_OUTER_FOCAL_GAMMA}" ]]; then
  COMMAND+=(--bands-outer-focal-gamma "${BANDS_OUTER_FOCAL_GAMMA}")
fi
if [[ -n "${EQUIVARIANCE_WEIGHT}" ]]; then
  COMMAND+=(--equivariance-weight "${EQUIVARIANCE_WEIGHT}")
fi
if [[ -n "${BANDS_WEIGHT}" ]]; then
  COMMAND+=(--bands-weight "${BANDS_WEIGHT}")
fi
if [[ -n "${BANDS_CALIBRATION_JSON}" ]]; then
  COMMAND+=(--bands-calibration-json "${BANDS_CALIBRATION_JSON}")
fi
if [[ -n "${ONECUT_WEIGHT}" ]]; then
  COMMAND+=(--onecut-weight "${ONECUT_WEIGHT}")
fi
if [[ -n "${ONECUT_CALIBRATION_JSON}" ]]; then
  COMMAND+=(--onecut-calibration-json "${ONECUT_CALIBRATION_JSON}")
fi
if [[ "${RESIZE}" == "1" ]]; then
  COMMAND+=(--resize)
fi
if [[ "${USE_AMP}" == "1" ]]; then
  COMMAND+=(--amp)
else
  COMMAND+=(--no-amp)
fi
if [[ "${USE_TELEMETRY}" == "1" ]]; then
  read -r -a TELEMETRY_PROBE_EPOCHS_ARRAY <<< "${TELEMETRY_PROBE_EPOCHS}"
  COMMAND+=(
    --telemetry
    --telemetry-probe-epochs "${TELEMETRY_PROBE_EPOCHS_ARRAY[@]}"
    --telemetry-probe-cases "${TELEMETRY_PROBE_CASES}"
    --telemetry-spatial-cases "${TELEMETRY_SPATIAL_CASES}"
  )
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
