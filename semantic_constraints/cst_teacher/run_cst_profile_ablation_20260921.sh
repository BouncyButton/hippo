#!/usr/bin/env bash
# One allocation runs all profile-head variants and seeds sequentially.
#SBATCH --job-name=cst-profile-msd
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_profile_ablation_${SLURM_JOB_ID:-local}}"
INFERENCE_DIR="${CST_INFERENCE_DIR:-${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236}"
VARIANTS=(sparse12_l1 dense32_smooth dense32_l1)
SEEDS=(0 1 2)

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] run_root=${RUN_ROOT}"
find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"
python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"

for VARIANT in "${VARIANTS[@]}"; do
  case "${VARIANT}" in
    sparse12_l1)
      SET_SIZE=12
      PROFILE_LOSS=l1
      BATCH_SIZE=32
      ;;
    dense32_smooth)
      SET_SIZE=32
      PROFILE_LOSS=smooth_l1
      BATCH_SIZE=16
      ;;
    dense32_l1)
      SET_SIZE=32
      PROFILE_LOSS=l1
      BATCH_SIZE=16
      ;;
    *)
      echo "[ERROR] unknown variant ${VARIANT}" >&2
      exit 1
      ;;
  esac

  for SEED in "${SEEDS[@]}"; do
    SEED_ROOT="${RUN_ROOT}/${VARIANT}_seed_${SEED}"
    mkdir -p "${SEED_ROOT}"
    echo "[STAGE] variant=${VARIANT} seed=${SEED} starts $(date -Is)"
    python -m semantic_constraints.cst_teacher.train_teacher \
      --device cuda \
      --fold 0 \
      --seed "${SEED}" \
      --epochs 150 \
      --patience 20 \
      --minimum-improvement 1e-4 \
      --batch-size "${BATCH_SIZE}" \
      --num-workers 0 \
      --channels 16 32 64 \
      --heads 4 \
      --set-size "${SET_SIZE}" \
      --minimum-set-size "${SET_SIZE}" \
      --slab-depth 3 \
      --inplane-size 32 \
      --descriptor-weight 0.25 \
      --profile-weight 4.0 \
      --profile-loss "${PROFILE_LOSS}" \
      --anomaly-weight 0.0 \
      --learning-rate 3e-4 \
      --weight-decay 1e-4 \
      --output-dir "${SEED_ROOT}" \
      2>&1 | tee "${SEED_ROOT}/train.log"

    python -m semantic_constraints.cst_teacher.evaluate_predictions \
      --checkpoint "${SEED_ROOT}/best_teacher.pt" \
      --inference-dir "${INFERENCE_DIR}" \
      --device cuda \
      --batch-size "${BATCH_SIZE}" \
      --output "${SEED_ROOT}/prediction_evaluation.json" \
      > "${SEED_ROOT}/prediction_evaluation.log"
  done
done

python -m semantic_constraints.cst_teacher.summarize_profile_ablation \
  --run-root "${RUN_ROOT}" \
  --variants "${VARIANTS[@]}" \
  --seeds "${SEEDS[@]}" \
  > "${RUN_ROOT}/profile_ablation_summary.log"

echo "[DONE] $(date -Is)"
cat "${RUN_ROOT}/profile_ablation_summary.md"
