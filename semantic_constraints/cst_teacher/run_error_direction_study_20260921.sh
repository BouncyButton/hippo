#!/usr/bin/env bash
# Patient-grouped missing-versus-extra direction probe, all six teacher banks.
#SBATCH --job-name=cst-error-direction
#SBATCH --partition=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
TYPE_ROOT="${CST_ERROR_TYPE_ROOT:-${RUN_BASE}/cst_error_types_665517}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_error_direction_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for SEED in 0 1 2; do
  for FOLD in 0 1; do
    python -m semantic_constraints.cst_teacher.probe_error_direction \
      --features "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/risk/risk_features_seed_${SEED}.npz" \
      --error-types "${TYPE_ROOT}/fold${FOLD}.npz" \
      --qc-oof-csv "${RUN_BASE}/cst_slice_qc_665471/seed_${SEED}/oof_predictions.csv" \
      --fold "${FOLD}" \
      --output "${RUN_ROOT}/seed_${SEED}_fold${FOLD}.json"
  done
done

python -m semantic_constraints.cst_teacher.summarize_error_direction_study --run-root "${RUN_ROOT}"
echo "[DONE] $(date -Is)"
