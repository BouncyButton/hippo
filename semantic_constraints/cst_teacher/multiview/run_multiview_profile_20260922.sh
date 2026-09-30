#!/usr/bin/env bash
# Matched MRI-only profile teachers: 3 views and 2 tri-view set budgets.
#SBATCH --job-name=cst-multiview-profile
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_multiview_profile_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
read -r -a FOLDS <<< "${CST_FOLDS:-0 1}"
read -r -a VIEWS <<< "${CST_VIEWS:-coronal32 sagittal32 axial32 triview32 triview96}"

find "${REPO_ROOT}/semantic_constraints/cst_teacher/multiview" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"
for FOLD in "${FOLDS[@]}"; do
  echo "[FOLD] ${FOLD} begins $(date -Is)"
  python -m semantic_constraints.cst_teacher.multiview.train_profile_study \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" \
    --inference-dir "${RUN_BASE}/cst_early_stopped_reanalysis_665422/fold${FOLD}/inference" \
    --output-dir "${RUN_ROOT}/fold${FOLD}" \
    --views "${VIEWS[@]}" \
    --max-epochs "${CST_MAX_EPOCHS:-100}" \
    --patience "${CST_PATIENCE:-12}" \
    --device cuda
done

echo "[DONE] $(date -Is)"
