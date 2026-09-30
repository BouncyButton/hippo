#!/usr/bin/env bash
# One GPU job runs both MSD folds; smoke/full selected by CST_STUDY_MODE.
#SBATCH --job-name=cst-local-correct
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
MODE="${CST_STUDY_MODE:-full}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_local_correction_${MODE}_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

case "${MODE}" in
  smoke)
    MAX_CASES=8
    OUTER_FOLDS=2
    MAX_EPOCHS=1
    VARIANTS=(cst)
    FOLDS=(0)
    ;;
  full)
    MAX_CASES=0
    OUTER_FOLDS=5
    MAX_EPOCHS=12
    VARIANTS=(image_mask cst)
    FOLDS=(0 1)
    ;;
  *) echo "Unknown CST_STUDY_MODE=${MODE}" >&2; exit 2 ;;
esac

python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"
for FOLD in "${FOLDS[@]}"; do
  python -m semantic_constraints.cst_teacher.run_local_correction_study \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" \
    --inference-dir "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/inference" \
    --features "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/risk/risk_features_seed_0.npz" \
    --output-dir "${RUN_ROOT}/fold${FOLD}" \
    --device cuda \
    --max-val-cases "${MAX_CASES}" \
    --outer-folds "${OUTER_FOLDS}" \
    --max-epochs "${MAX_EPOCHS}" \
    --variants "${VARIANTS[@]}"
done

echo "[DONE] $(date -Is)"
