#!/usr/bin/env bash
# Frozen fold-1 replication: Swin training/inference, three CST teachers,
# repeated grouped risk probes, and selective-QC analysis in one allocation.
#SBATCH --job-name=cst-fold1-repl
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
ENV_NAME="${CST_ENV_NAME:-hippocampus}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_fold1_replication_${SLURM_JOB_ID:-local}}"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"
MODEL_ROOT="${RUN_ROOT}/swin_model"
CHECKPOINT="${MODEL_ROOT}/MSD_fold1/model.pt"
INFERENCE_DIR="${RUN_ROOT}/swin_inference"
SEEDS=(0 1 2)

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] run_root=${RUN_ROOT}"
find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/cst_source_sha256.txt"
sha256sum \
  "${REPO_ROOT}/baselines/swin_unetr/swin_unetr.py" \
  "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
  > "${RUN_ROOT}/swin_source_sha256.txt"
python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher" "${REPO_ROOT}/baselines/swin_unetr"

python - <<'PY'
import json
from pathlib import Path
import torch

splits = json.loads(Path("datasets/Dataset101_MSD/splits_final.json").read_text())
fold = splits[1]
assert len(fold["train"]) == 208 and len(fold["val"]) == 52
assert not (set(fold["train"]) & set(fold["val"]))
assert torch.cuda.is_available()
print({"fold": 1, "train": 208, "validation": 52, "gpu": torch.cuda.get_device_name(0)})
PY

echo "[STAGE] Swin fold-1 training starts $(date -Is)"
python "${REPO_ROOT}/baselines/swin_unetr/swin_unetr.py" \
  --pkl "${PKL}" \
  --dataset MSD \
  --splits-json "${SPLITS}" \
  --fold 1 \
  --epochs 50 \
  --batch-size 2 \
  --spatial-size 64 64 64 \
  --optim-mode adamw_0.01 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --step-size 20 \
  --adamw-gamma 0.5 \
  --model-out-dir "${MODEL_ROOT}" \
  --no-wandb \
  2>&1 | tee "${RUN_ROOT}/swin_train.log"
test -f "${CHECKPOINT}"

echo "[STAGE] Swin fold-1 inference starts $(date -Is)"
python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
  --checkpoint "${CHECKPOINT}" \
  --pkl "${PKL}" \
  --dataset MSD \
  --splits-json "${SPLITS}" \
  --fold 1 \
  --split val \
  --batch-size 2 \
  --spatial-size 64 64 64 \
  --output-dir "${INFERENCE_DIR}" \
  2>&1 | tee "${RUN_ROOT}/swin_inference.log"
test -f "${INFERENCE_DIR}/prediction_index.csv"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${RUN_ROOT}/dense32_smooth_seed_${SEED}"
  mkdir -p "${SEED_ROOT}"
  echo "[STAGE] CST fold-1 seed=${SEED} starts $(date -Is)"
  python -m semantic_constraints.cst_teacher.train_teacher \
    --device cuda \
    --fold 1 \
    --seed "${SEED}" \
    --epochs 150 \
    --patience 20 \
    --minimum-improvement 1e-4 \
    --batch-size 16 \
    --num-workers 0 \
    --channels 16 32 64 \
    --heads 4 \
    --set-size 32 \
    --minimum-set-size 32 \
    --slab-depth 3 \
    --inplane-size 32 \
    --descriptor-weight 0.25 \
    --profile-weight 4.0 \
    --profile-loss smooth_l1 \
    --anomaly-weight 0.0 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --output-dir "${SEED_ROOT}" \
    2>&1 | tee "${SEED_ROOT}/train.log"

  python -m semantic_constraints.cst_teacher.evaluate_predictions \
    --checkpoint "${SEED_ROOT}/best_teacher.pt" \
    --inference-dir "${INFERENCE_DIR}" \
    --device cuda \
    --batch-size 16 \
    --output "${SEED_ROOT}/prediction_evaluation.json" \
    > "${SEED_ROOT}/prediction_evaluation.log"
done

python -m semantic_constraints.cst_teacher.summarize_profile_ablation \
  --run-root "${RUN_ROOT}" \
  --variants dense32_smooth \
  --seeds "${SEEDS[@]}" \
  > "${RUN_ROOT}/profile_ablation_summary.log"

mkdir -p "${RUN_ROOT}/risk_probe"
CHECKPOINTS=()
for SEED in "${SEEDS[@]}"; do
  CHECKPOINTS+=("${RUN_ROOT}/dense32_smooth_seed_${SEED}/best_teacher.pt")
done
python -m semantic_constraints.cst_teacher.risk_probe \
  --checkpoints "${CHECKPOINTS[@]}" \
  --inference-dir "${INFERENCE_DIR}" \
  --output-dir "${RUN_ROOT}/risk_probe" \
  --device cuda \
  --batch-size 16 \
  --outer-folds 5 \
  --inner-folds 4 \
  --repeats 10 \
  --alphas 0.0001 0.001 0.01 0.1 1 10 100 \
  2>&1 | tee "${RUN_ROOT}/risk_probe.log"

python -m semantic_constraints.cst_teacher.summarize_risk_utility \
  --risk-report "${RUN_ROOT}/risk_probe/risk_probe_report.json" \
  --features "${RUN_ROOT}/risk_probe/risk_features_seed_0.npz" \
  --output "${RUN_ROOT}/risk_utility.json" \
  > "${RUN_ROOT}/risk_utility.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/profile_ablation_summary.md"
cat "${RUN_ROOT}/risk_probe/risk_probe_report.md"
cat "${RUN_ROOT}/risk_utility.md"
