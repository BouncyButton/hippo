#!/usr/bin/env bash
# Audit the completed ranker's reliance on MRI and mask inputs.
#SBATCH --job-name=cst-candidate-dependence
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
CANDIDATE_ROOT="${CST_CANDIDATE_ROOT:?Set CST_CANDIDATE_ROOT to the completed ranker run directory}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_candidate_dependence_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
for FOLD in 0 1; do
  python -m semantic_constraints.cst_teacher.multiview.audit_candidate_dependence \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" \
    --inference-dir "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/inference" \
    --checkpoint-dir "${CANDIDATE_ROOT}/fold${FOLD}" \
    --output "${RUN_ROOT}/fold${FOLD}.json" \
    --device cuda
done
