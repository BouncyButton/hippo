#!/usr/bin/env bash
# Frozen train-split inference and overfitting audit for Swin folds 0 and 1.
#SBATCH --job-name=swin-overfit-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${CST_ENV_NAME:-hippocampus}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/swin_overfit_audit_${SLURM_JOB_ID:-local}}"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"
FOLD0_ROOT="${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0"
FOLD1_ROOT="${RUN_BASE}/cst_fold1_replication_665384"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FOLD0_CHECKPOINT="${FOLD0_ROOT}/model.pt"
FOLD1_CHECKPOINT="${FOLD1_ROOT}/swin_model/MSD_fold1/model.pt"
FOLD0_VAL="${FOLD0_ROOT}/inference_fold0_val_600236/metrics_summary.json"
FOLD1_VAL="${FOLD1_ROOT}/swin_inference/metrics_summary.json"
test -f "${FOLD0_CHECKPOINT}"
test -f "${FOLD1_CHECKPOINT}"
test -f "${FOLD0_VAL}"
test -f "${FOLD1_VAL}"

for FOLD in 0 1; do
  if [[ "${FOLD}" == "0" ]]; then
    CHECKPOINT="${FOLD0_CHECKPOINT}"
  else
    CHECKPOINT="${FOLD1_CHECKPOINT}"
  fi
  OUTPUT="${RUN_ROOT}/fold${FOLD}_train_inference"
  echo "[STAGE] fold=${FOLD} train inference starts $(date -Is)"
  python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
    --checkpoint "${CHECKPOINT}" \
    --pkl "${PKL}" \
    --dataset MSD \
    --splits-json "${SPLITS}" \
    --fold "${FOLD}" \
    --split train \
    --batch-size 2 \
    --spatial-size 64 64 64 \
    --output-dir "${OUTPUT}" \
    2>&1 | tee "${RUN_ROOT}/fold${FOLD}_train_inference.log"
done

python -m semantic_constraints.cst_teacher.audit_swin_overfit \
  --fold0-train-metrics "${RUN_ROOT}/fold0_train_inference/metrics_summary.json" \
  --fold0-val-metrics "${FOLD0_VAL}" \
  --fold0-train-log "${REPO_ROOT}/slurm-swinunetr-600169.out" \
  --fold1-train-metrics "${RUN_ROOT}/fold1_train_inference/metrics_summary.json" \
  --fold1-val-metrics "${FOLD1_VAL}" \
  --fold1-train-log "${FOLD1_ROOT}/swin_train.log" \
  --splits-json "${SPLITS}" \
  --output "${RUN_ROOT}/overfit_audit.json" \
  2>&1 | tee "${RUN_ROOT}/overfit_audit.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/overfit_audit.md"
