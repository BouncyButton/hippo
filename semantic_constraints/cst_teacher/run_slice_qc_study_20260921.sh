#!/usr/bin/env bash
# Three-teacher-seed study of trainable CST-conditioned slice-quality heads.
#SBATCH --job-name=cst-slice-qc
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
FEATURE_ROOT="${RUN_BASE}/cst_early_stopped_reanalysis_665422"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_slice_qc_${SLURM_JOB_ID:-local}}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"
python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"

for SEED in 0 1 2; do
  FOLD0="${FEATURE_ROOT}/fold0/risk/risk_features_seed_${SEED}.npz"
  FOLD1="${FEATURE_ROOT}/fold1/risk/risk_features_seed_${SEED}.npz"
  test -f "${FOLD0}"
  test -f "${FOLD1}"
  OUTPUT="${RUN_ROOT}/seed_${SEED}"
  echo "[STAGE] seed=${SEED} patient-grouped head study starts $(date -Is)"
  python -m semantic_constraints.cst_teacher.run_slice_qc_experiment \
    --fold0-features "${FOLD0}" \
    --fold1-features "${FOLD1}" \
    --output-dir "${OUTPUT}" \
    --device cpu \
    --max-epochs 80 \
    --patience 10 \
    --outer-folds 5 \
    --seed 20260921 \
    2>&1 | tee "${OUTPUT}.log"

  python -m semantic_constraints.cst_teacher.score_slice_qc \
    --model "${OUTPUT}/models/fold0/ridge_portable.npz" \
    --features "${FOLD1}" \
    --output "${OUTPUT}/unlabeled_scoring_smoke.csv" \
    2>&1 | tee "${OUTPUT}/scoring_smoke.log"
done

python -m semantic_constraints.cst_teacher.summarize_slice_qc_study \
  --run-root "${RUN_ROOT}" \
  --seeds 0 1 2 \
  2>&1 | tee "${RUN_ROOT}/study_summary.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/study_summary.md"
