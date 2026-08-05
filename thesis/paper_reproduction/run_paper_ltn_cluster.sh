#!/usr/bin/env bash
# One run:
# sbatch thesis/paper_reproduction/run_paper_ltn_cluster.sh --method baseline --fold 1 --train-fraction 1.0
# Full paper matrix (30 jobs):
# sbatch --array=0-29 thesis/paper_reproduction/run_paper_ltn_cluster.sh --matrix
# Epsilon sweep (five epsilon values x five folds):
# sbatch --array=0-24 thesis/paper_reproduction/run_paper_ltn_cluster.sh --epsilon-matrix
#SBATCH --job-name=paper-ltn
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch thesis/paper_reproduction/run_paper_ltn_cluster.sh --method baseline|ltn --fold 1..5 --train-fraction 1.0|0.25|0.05 [options]
  sbatch --array=0-29 thesis/paper_reproduction/run_paper_ltn_cluster.sh --matrix [options]
  sbatch --array=0-24 thesis/paper_reproduction/run_paper_ltn_cluster.sh --epsilon-matrix [options]

Options:
  --data-root PATH       Parent directory containing Task04_Hippocampus (default: thesis/data)
  --method METHOD        baseline or ltn (required unless --matrix)
  --fold N               One-based notebook KFold index (required unless --matrix)
  --train-fraction F     One of 1.0, 0.25, 0.05 (required unless --matrix)
  --matrix               Map SLURM_ARRAY_TASK_ID 0..29 to all paper fractions/methods/folds
  --epsilon-matrix       Map array tasks to epsilon values x five LTN folds
  --epsilon-values CSV   Sweep values (default: 0,500,1000,2500,5000)
  --volume-epsilon E     Volume-gap tolerance for one run (default: 5000)
  --volume-grounding G   paper-hard or soft-probability (default: paper-hard)
  --prepare-data         Download Decathlon data once, then exit; do not run this in an array
  --expected-samples N   Effective DecathlonDataset count after its 20% split (default: 208)
  --resume                Resume an interrupted run in its stable output directory
  --wandb                 Log each run and final checkpoint to Weights & Biases
  --wandb-project NAME    W&B project (default: hippopotamus-project)
  --wandb-entity NAME     W&B entity (default: focacciafilippo-bocconi-university)
  --env-name NAME        Conda environment (default: hippocampus)
  --output-root PATH     Parent directory for reproducible run artifacts
EOF
}

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/thesis/paper_reproduction/train_paper_ltn.py" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "${PWD}/thesis/paper_reproduction/train_paper_ltn.py" ]]; then
  # Supports remote `sbatch --chdir=REPO_ROOT ...`, where SLURM_SUBMIT_DIR
  # still points to the SSH login directory rather than the job's cwd.
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
DATA_ROOT="${REPO_ROOT}/thesis/data"
OUTPUT_ROOT="${REPO_ROOT}/thesis/runs/paper_reproduction"
ENV_NAME="hippocampus"
METHOD=""
FOLD=""
FRACTION=""
MATRIX="0"
EPSILON_MATRIX="0"
EPSILON_VALUES="0,500,1000,2500,5000"
VOLUME_EPSILON="5000"
VOLUME_GROUNDING="paper-hard"
DOWNLOAD="0"
PREPARE_DATA="0"
EXPECTED_SAMPLES="208"
RESUME="0"
USE_WANDB="0"
WANDB_PROJECT="hippopotamus-project"
WANDB_ENTITY="focacciafilippo-bocconi-university"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --train-fraction) FRACTION="$2"; shift 2 ;;
    --matrix) MATRIX="1"; shift ;;
    --epsilon-matrix) EPSILON_MATRIX="1"; shift ;;
    --epsilon-values) EPSILON_VALUES="$2"; shift 2 ;;
    --volume-epsilon) VOLUME_EPSILON="$2"; shift 2 ;;
    --volume-grounding) VOLUME_GROUNDING="$2"; shift 2 ;;
    --prepare-data) PREPARE_DATA="1"; shift ;;
    --expected-samples) EXPECTED_SAMPLES="$2"; shift 2 ;;
    --resume) RESUME="1"; shift ;;
    --wandb) USE_WANDB="1"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${PREPARE_DATA}" == "1" && ("${MATRIX}" == "1" || "${EPSILON_MATRIX}" == "1") ]]; then
  echo "[ERROR] --prepare-data cannot be combined with an experiment matrix." >&2
  exit 2
fi
if [[ "${MATRIX}" == "1" && "${EPSILON_MATRIX}" == "1" ]]; then
  echo "[ERROR] --matrix and --epsilon-matrix are mutually exclusive." >&2
  exit 2
fi
if [[ "${MATRIX}" == "1" ]]; then
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -gt 29 ]]; then
    echo "[ERROR] --matrix requires sbatch --array=0-29." >&2
    exit 2
  fi
  FRACTIONS=("1.0" "0.25" "0.05")
  FRACTION="${FRACTIONS[$((SLURM_ARRAY_TASK_ID / 10))]}"
  METHOD=$([[ $(((SLURM_ARRAY_TASK_ID % 10) / 5)) -eq 0 ]] && echo baseline || echo ltn)
  FOLD="$((SLURM_ARRAY_TASK_ID % 5 + 1))"
fi
if [[ "${EPSILON_MATRIX}" == "1" ]]; then
  IFS=',' read -r -a EPSILONS <<< "${EPSILON_VALUES}"
  TASK_COUNT="$((${#EPSILONS[@]} * 5))"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${TASK_COUNT}" ]]; then
    echo "[ERROR] --epsilon-matrix requires array indices 0..$((TASK_COUNT - 1)) for ${#EPSILONS[@]} epsilon values." >&2
    exit 2
  fi
  VOLUME_EPSILON="${EPSILONS[$((SLURM_ARRAY_TASK_ID / 5))]}"
  METHOD="ltn"
  FOLD="$((SLURM_ARRAY_TASK_ID % 5 + 1))"
  FRACTION="${FRACTION:-0.05}"
fi

if [[ "${PREPARE_DATA}" != "1" && (! "${METHOD}" =~ ^(baseline|ltn)$ || ! "${FOLD}" =~ ^[1-5]$ || ! "${FRACTION}" =~ ^(1\.0|0\.25|0\.05)$) ]]; then
  echo "[ERROR] Provide --method baseline|ltn, --fold 1..5, and --train-fraction 1.0|0.25|0.05." >&2
  exit 2
fi
if [[ ! "${VOLUME_EPSILON}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] --volume-epsilon and all --epsilon-values must be non-negative numbers." >&2
  exit 2
fi
if [[ ! "${VOLUME_GROUNDING}" =~ ^(paper-hard|soft-probability)$ ]]; then
  echo "[ERROR] --volume-grounding must be paper-hard or soft-probability." >&2
  exit 2
fi
if [[ "${PREPARE_DATA}" != "1" && ! -d "${DATA_ROOT}/Task04_Hippocampus" ]]; then
  echo "[ERROR] Decathlon dataset is missing under ${DATA_ROOT}. Run one preparation job with --prepare-data first." >&2
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
python -c 'import ltn, monai, sklearn, torch; print("[INFO] torch={}, monai={}, ltn={}, sklearn={}, cuda={}".format(torch.__version__, monai.__version__, getattr(ltn, "__version__", "unknown"), sklearn.__version__, torch.cuda.is_available()))'

if [[ "${PREPARE_DATA}" == "1" ]]; then
  python "${REPO_ROOT}/thesis/paper_reproduction/prepare_decathlon.py" --data-root "${DATA_ROOT}" --expected-samples "${EXPECTED_SAMPLES}"
  exit 0
fi

if [[ "${EPSILON_MATRIX}" == "1" || "${VOLUME_EPSILON}" != "5000" || "${VOLUME_GROUNDING}" != "paper-hard" ]]; then
  RUN_NAME="epsilon_${VOLUME_EPSILON}_grounding_${VOLUME_GROUNDING}_fraction_${FRACTION}_method_${METHOD}_fold_${FOLD}"
else
  RUN_NAME="fraction_${FRACTION}_method_${METHOD}_fold_${FOLD}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
CMD=(
  python "${REPO_ROOT}/thesis/paper_reproduction/train_paper_ltn.py"
  --data-root "${DATA_ROOT}"
  --method "${METHOD}"
  --fold "${FOLD}"
  --train-fraction "${FRACTION}"
  --volume-epsilon "${VOLUME_EPSILON}"
  --volume-grounding "${VOLUME_GROUNDING}"
  --expected-samples "${EXPECTED_SAMPLES}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
)
if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  CMD+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-entity "${WANDB_ENTITY}")
fi
echo "[INFO] ${CMD[*]}"
"${CMD[@]}"
