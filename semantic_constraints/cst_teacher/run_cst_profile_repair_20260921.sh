#!/usr/bin/env bash
# One allocation runs all dense-profile counterfactual configurations.
#SBATCH --job-name=cst-profile-repair
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${CST_ENV_NAME:-hippocampus}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
PROFILE_ROOT="${CST_PROFILE_ROOT:-${RUN_BASE}/cst_profile_ablation_665313}"
PROFILE_VARIANT="${CST_PROFILE_VARIANT:-dense32_smooth}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_profile_repair_${SLURM_JOB_ID:-local}}"
INFERENCE_DIR="${CST_INFERENCE_DIR:-${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

CHECKPOINTS=()
for SEED in 0 1 2; do
  CHECKPOINT="${PROFILE_ROOT}/${PROFILE_VARIANT}_seed_${SEED}/best_teacher.pt"
  test -f "${CHECKPOINT}"
  CHECKPOINTS+=("${CHECKPOINT}")
done

python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"
python -m semantic_constraints.cst_teacher.profile_counterfactual \
  --checkpoints "${CHECKPOINTS[@]}" \
  --inference-dir "${INFERENCE_DIR}" \
  --output "${RUN_ROOT}/profile_counterfactual.json" \
  --device cuda \
  --batch-size 16 \
  --calibration-percentile 50 \
  --minimum-teacher-votes 2 \
  --modes slice_ap_bias slice_fg_ap_bias \
  --gammas 0.001 0.01 0.1 1.0 \
  --steps 100 \
  --learning-rate 0.1 \
  2>&1 | tee "${RUN_ROOT}/profile_counterfactual.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/profile_counterfactual.md"
