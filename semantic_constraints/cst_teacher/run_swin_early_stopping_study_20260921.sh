#!/usr/bin/env bash
# Matched two-fold SwinUNETR early-stopping study.
#SBATCH --job-name=swin-early-stop
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${CST_ENV_NAME:-hippocampus}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/swin_early_stopping_${SLURM_JOB_ID:-local}}"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}/models" "${RUN_ROOT}/inference"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for FOLD in 0 1; do
  echo "[STAGE] fold=${FOLD} early-stopped training starts $(date -Is)"
  python "${REPO_ROOT}/baselines/swin_unetr/swin_unetr.py" \
    --pkl "${PKL}" \
    --dataset MSD \
    --fold "${FOLD}" \
    --epochs 50 \
    --batch-size 2 \
    --spatial-size 64 64 64 \
    --splits-json "${SPLITS}" \
    --optim-mode adamw_0.01 \
    --learning-rate 0.0001 \
    --weight-decay 0.00001 \
    --step-size 20 \
    --adamw-gamma 0.5 \
    --model-out-dir "${RUN_ROOT}/models" \
    --save-best \
    --early-stopping-patience 8 \
    --early-stopping-min-delta 0 \
    --no-wandb \
    2>&1 | tee "${RUN_ROOT}/fold${FOLD}_train.log"

  for CHECKPOINT in best stopped; do
    if [[ "${CHECKPOINT}" == "best" ]]; then
      MODEL="${RUN_ROOT}/models/MSD_fold${FOLD}/model_best.pt"
    else
      MODEL="${RUN_ROOT}/models/MSD_fold${FOLD}/model.pt"
    fi
    for SPLIT in train val; do
      OUTPUT="${RUN_ROOT}/inference/fold${FOLD}_${CHECKPOINT}_${SPLIT}"
      echo "[STAGE] fold=${FOLD} checkpoint=${CHECKPOINT} split=${SPLIT} starts $(date -Is)"
      python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
        --checkpoint "${MODEL}" \
        --pkl "${PKL}" \
        --dataset MSD \
        --splits-json "${SPLITS}" \
        --fold "${FOLD}" \
        --split "${SPLIT}" \
        --batch-size 2 \
        --spatial-size 64 64 64 \
        --metrics-only \
        --output-dir "${OUTPUT}" \
        2>&1 | tee "${RUN_ROOT}/fold${FOLD}_${CHECKPOINT}_${SPLIT}.log"
    done
  done
done

python -m semantic_constraints.cst_teacher.summarize_swin_early_stopping \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/early_stopping_summary.json" \
  2>&1 | tee "${RUN_ROOT}/early_stopping_summary.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/early_stopping_summary.md"
