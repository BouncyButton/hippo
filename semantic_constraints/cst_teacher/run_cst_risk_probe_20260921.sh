#!/usr/bin/env bash
# One allocation extracts all seeds and runs repeated nested grouped probes.
#SBATCH --job-name=cst-risk-probe
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
PROFILE_ROOT="${CST_PROFILE_ROOT:-${RUN_BASE}/cst_profile_ablation_665313}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_risk_probe_${SLURM_JOB_ID:-local}}"
INFERENCE_DIR="${CST_INFERENCE_DIR:-${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CHECKPOINTS=()
for SEED in 0 1 2; do
  CHECKPOINT="${PROFILE_ROOT}/dense32_smooth_seed_${SEED}/best_teacher.pt"
  test -f "${CHECKPOINT}"
  CHECKPOINTS+=("${CHECKPOINT}")
done

find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"
python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"
python -m semantic_constraints.cst_teacher.risk_probe \
  --checkpoints "${CHECKPOINTS[@]}" \
  --inference-dir "${INFERENCE_DIR}" \
  --output-dir "${RUN_ROOT}" \
  --device cuda \
  --batch-size 16 \
  --outer-folds 5 \
  --inner-folds 4 \
  --repeats 10 \
  --alphas 0.0001 0.001 0.01 0.1 1 10 100 \
  2>&1 | tee "${RUN_ROOT}/risk_probe.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/risk_probe_report.md"
