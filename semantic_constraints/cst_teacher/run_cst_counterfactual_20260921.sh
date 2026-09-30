#!/usr/bin/env bash
# One allocation runs the complete multi-mode/multi-strength repair sweep.
#SBATCH --job-name=cst-repair-msd
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

ENV_NAME="${CST_ENV_NAME:-hippocampus}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
TEACHER_ROOT="${CST_TEACHER_ROOT:-${RUN_BASE}/cst_teacher_fold0_multiseed_665270}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_counterfactual_${SLURM_JOB_ID:-local}}"
INFERENCE_DIR="${CST_INFERENCE_DIR:-${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

CHECKPOINTS=()
for SEED in 0 1 2; do
  CHECKPOINT="${TEACHER_ROOT}/seed_${SEED}/best_teacher.pt"
  test -f "${CHECKPOINT}"
  CHECKPOINTS+=("${CHECKPOINT}")
done

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] teacher_root=${TEACHER_ROOT}"
echo "[INFO] run_root=${RUN_ROOT}"
sha256sum "${CHECKPOINTS[@]}" > "${RUN_ROOT}/checkpoint_sha256.txt"
sha256sum "${REPO_ROOT}/semantic_constraints/cst_teacher/counterfactual_repair.py" > "${RUN_ROOT}/source_sha256.txt"

python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"
python -m semantic_constraints.cst_teacher.counterfactual_repair \
  --checkpoints "${CHECKPOINTS[@]}" \
  --inference-dir "${INFERENCE_DIR}" \
  --output "${RUN_ROOT}/counterfactual_repair.json" \
  --device cuda \
  --batch-size 32 \
  --minimum-teacher-votes 2 \
  --modes global_ap_bias voxel_ap_field \
  --gammas 0.001 0.01 0.1 1.0 10.0 \
  --steps 100 \
  --learning-rate 0.1 \
  2>&1 | tee "${RUN_ROOT}/counterfactual_repair.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/counterfactual_repair.md"
