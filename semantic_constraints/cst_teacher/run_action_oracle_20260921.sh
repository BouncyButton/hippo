#!/usr/bin/env bash
# Diagnostic upper bound for simple log-odds edits, two folds x three QC seeds.
#SBATCH --job-name=cst-action-oracle
#SBATCH --partition=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_action_oracle_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for FOLD in 0 1; do
  python -m semantic_constraints.cst_teacher.probe_action_oracle \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" \
    --inference-dir "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/inference" \
    --reference-features "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/risk/risk_features_seed_0.npz" \
    --error-types "${RUN_BASE}/cst_error_types_665517/fold${FOLD}.npz" \
    --qc-oof-csv "${RUN_BASE}/cst_slice_qc_665471/seed_0/oof_predictions.csv" \
                 "${RUN_BASE}/cst_slice_qc_665471/seed_1/oof_predictions.csv" \
                 "${RUN_BASE}/cst_slice_qc_665471/seed_2/oof_predictions.csv" \
    --output "${RUN_ROOT}/fold${FOLD}.json"
done

echo "[DONE] $(date -Is)"
