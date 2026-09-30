#!/usr/bin/env bash
# Re-evaluate frozen CST teachers on best early-stopped Swin checkpoints.
#SBATCH --job-name=cst-early-compare
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
STUDY_ROOT="${RUN_BASE}/swin_early_stopping_665420"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_early_stopped_reanalysis_${SLURM_JOB_ID:-local}}"
FOLD0_TEACHERS="${RUN_BASE}/cst_profile_ablation_665313"
FOLD1_TEACHERS="${RUN_BASE}/cst_fold1_replication_665384"
FOLD0_OLD_RISK="${RUN_BASE}/cst_risk_probe_665373"
FOLD1_OLD_RISK="${FOLD1_TEACHERS}/risk_probe"
PKL="${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
SPLITS="${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

test -f "${STUDY_ROOT}/early_stopping_summary.json"
test -f "${FOLD0_OLD_RISK}/risk_probe_report.json"
test -f "${FOLD1_OLD_RISK}/risk_probe_report.json"
find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"

# Recreate the epoch-50 fold-0 QC summary from its already-completed probe.
python -m semantic_constraints.cst_teacher.summarize_risk_utility \
  --risk-report "${FOLD0_OLD_RISK}/risk_probe_report.json" \
  --features "${FOLD0_OLD_RISK}/risk_features_seed_0.npz" \
  --output "${RUN_ROOT}/fold0_epoch50_risk_utility.json" \
  > "${RUN_ROOT}/fold0_epoch50_risk_utility.log"

for FOLD in 0 1; do
  FOLD_ROOT="${RUN_ROOT}/fold${FOLD}"
  INFERENCE="${FOLD_ROOT}/inference"
  mkdir -p "${FOLD_ROOT}/teacher" "${FOLD_ROOT}/risk"
  CHECKPOINT="${STUDY_ROOT}/models/MSD_fold${FOLD}/model_best.pt"
  test -f "${CHECKPOINT}"
  echo "[STAGE] fold=${FOLD} best-checkpoint validation probabilities $(date -Is)"
  python "${REPO_ROOT}/baselines/swin_unetr/infer_swin_unetr.py" \
    --checkpoint "${CHECKPOINT}" \
    --pkl "${PKL}" \
    --dataset MSD \
    --splits-json "${SPLITS}" \
    --fold "${FOLD}" \
    --split val \
    --batch-size 2 \
    --spatial-size 64 64 64 \
    --output-dir "${INFERENCE}" \
    2>&1 | tee "${FOLD_ROOT}/inference.log"

  CHECKPOINTS=()
  for SEED in 0 1 2; do
    if [[ "${FOLD}" == "0" ]]; then
      TEACHER="${FOLD0_TEACHERS}/dense32_smooth_seed_${SEED}/best_teacher.pt"
    else
      TEACHER="${FOLD1_TEACHERS}/dense32_smooth_seed_${SEED}/best_teacher.pt"
    fi
    test -f "${TEACHER}"
    CHECKPOINTS+=("${TEACHER}")
    OUTPUT="${FOLD_ROOT}/teacher/seed_${SEED}"
    mkdir -p "${OUTPUT}"
    echo "[STAGE] fold=${FOLD} frozen CST seed=${SEED} $(date -Is)"
    python -m semantic_constraints.cst_teacher.evaluate_predictions \
      --checkpoint "${TEACHER}" \
      --inference-dir "${INFERENCE}" \
      --device cuda \
      --batch-size 16 \
      --output "${OUTPUT}/prediction_evaluation.json" \
      > "${OUTPUT}/prediction_evaluation.log"
  done

  echo "[STAGE] fold=${FOLD} grouped CST slice-risk probe $(date -Is)"
  python -m semantic_constraints.cst_teacher.risk_probe \
    --checkpoints "${CHECKPOINTS[@]}" \
    --inference-dir "${INFERENCE}" \
    --output-dir "${FOLD_ROOT}/risk" \
    --device cuda \
    --batch-size 16 \
    --outer-folds 5 \
    --inner-folds 4 \
    --repeats 10 \
    --alphas 0.0001 0.001 0.01 0.1 1 10 100 \
    2>&1 | tee "${FOLD_ROOT}/risk.log"

  python -m semantic_constraints.cst_teacher.summarize_risk_utility \
    --risk-report "${FOLD_ROOT}/risk/risk_probe_report.json" \
    --features "${FOLD_ROOT}/risk/risk_features_seed_0.npz" \
    --output "${FOLD_ROOT}/risk_utility.json" \
    > "${FOLD_ROOT}/risk_utility.log"
done

python -m semantic_constraints.cst_teacher.compare_swin_cst_checkpoints \
  --old-fold0-teachers "${FOLD0_TEACHERS}" \
  --old-fold0-risk "${FOLD0_OLD_RISK}" \
  --old-fold0-utility "${RUN_ROOT}/fold0_epoch50_risk_utility.json" \
  --old-fold1-teachers "${FOLD1_TEACHERS}" \
  --old-fold1-risk "${FOLD1_OLD_RISK}" \
  --old-fold1-utility "${FOLD1_TEACHERS}/risk_utility.json" \
  --new-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/checkpoint_comparison.json" \
  2>&1 | tee "${RUN_ROOT}/checkpoint_comparison.log"

echo "[DONE] $(date -Is)"
