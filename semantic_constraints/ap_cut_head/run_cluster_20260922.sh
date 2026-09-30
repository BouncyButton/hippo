#!/usr/bin/env bash
# Frozen-Swin cut-head experiment: both arms and both development folds in one job.
#SBATCH --job-name=ap-cut-head
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/ap_cut_head_${SLURM_JOB_ID:-local}}"
STUDY_ROOT="${RUN_BASE}/swin_early_stopping_665420"
EARLY_ROOT="${RUN_BASE}/cst_early_stopped_reanalysis_665422"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

find "${REPO_ROOT}/semantic_constraints/ap_cut_head" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"
for FOLD in 0 1; do
  CHECKPOINT="${STUDY_ROOT}/models/MSD_fold${FOLD}/model_best.pt"
  VAL_INFERENCE="${EARLY_ROOT}/fold${FOLD}/inference"
  TRAIN_INFERENCE="${RUN_ROOT}/fold${FOLD}/train_inference"
  test -f "${CHECKPOINT}"
  test -f "${VAL_INFERENCE}/prediction_index.csv"
  echo "[FOLD] ${FOLD} frozen train inference begins $(date -Is)"
  python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
    --checkpoint "${CHECKPOINT}" \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --dataset MSD \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" --split train --batch-size 2 \
    --spatial-size 64 64 64 --output-dir "${TRAIN_INFERENCE}" \
    > "${RUN_ROOT}/fold${FOLD}_train_inference.log" 2>&1

  echo "[FOLD] ${FOLD} cut head begins $(date -Is)"
  python -m semantic_constraints.ap_cut_head.train \
    --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
    --splits-json "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
    --fold "${FOLD}" --train-inference-dir "${TRAIN_INFERENCE}" \
    --val-inference-dir "${VAL_INFERENCE}" \
    --output-dir "${RUN_ROOT}/fold${FOLD}" \
    --max-epochs "${CUT_MAX_EPOCHS:-35}" --patience "${CUT_PATIENCE:-7}" \
    --device cuda > "${RUN_ROOT}/fold${FOLD}_cut_head.log" 2>&1
  echo "[FOLD] ${FOLD} completed $(date -Is)"
done
echo "[DONE] $(date -Is)"
