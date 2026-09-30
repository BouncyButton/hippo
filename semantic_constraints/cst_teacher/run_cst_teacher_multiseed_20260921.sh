#!/usr/bin/env bash
# One allocation runs all seeds and their dependent discovery analyses.
# Submit from the cluster checkout:
#   sbatch semantic_constraints/cst_teacher/run_cst_teacher_multiseed_20260921.sh
#SBATCH --job-name=cst-teacher-msd
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_teacher_fold0_multiseed_${SLURM_JOB_ID:-local}}"
INFERENCE_DIR="${CST_INFERENCE_DIR:-${REPO_ROOT}/models/swin_unetr/msd_fold0_20260722_190019_600169/MSD_fold0/inference_fold0_val_600236}"
SEEDS=(0 1 2)

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available" >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-cst-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "${MPLCONFIGDIR}"

echo "[INFO] host=$(hostname), job=${SLURM_JOB_ID:-local}, gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] run_root=${RUN_ROOT}"
git -C "${REPO_ROOT}" status --short > "${RUN_ROOT}/git_status.txt"
find "${REPO_ROOT}/semantic_constraints/cst_teacher" -maxdepth 1 -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/source_sha256.txt"

python - <<'PY'
import json
import torch
from pathlib import Path

split = json.loads(Path("datasets/Dataset101_MSD/splits_final.json").read_text())[0]
assert len(split["train"]) == 208 and len(split["val"]) == 52
assert not (set(split["train"]) & set(split["val"]))
assert torch.cuda.is_available(), "Slurm allocation has no visible CUDA device"
print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "train": 208, "validation": 52})
PY

if [[ ! -f "${INFERENCE_DIR}/prediction_index.csv" ]]; then
  echo "[ERROR] archived fold-0 inference probabilities are missing: ${INFERENCE_DIR}" >&2
  exit 1
fi

python -m compileall -q "${REPO_ROOT}/semantic_constraints/cst_teacher"
python - <<'PY' | tee "${RUN_ROOT}/source_smoke.txt"
import torch
from semantic_constraints.cst_teacher.model import CSTDescriptorTeacher

model = CSTDescriptorTeacher(channels=(8, 16), heads=2).cuda().eval()
with torch.no_grad():
    output = model(torch.randn(2, 6, 3, 16, 16, device="cuda"),
                   torch.randn(2, 6, 1, device="cuda"))
assert output.descriptor_quantiles.shape == (2, 4, 3)
assert output.slice_profiles.shape == (2, 6, 2)
print("cluster source/CUDA smoke passed")
PY

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  mkdir -p "${SEED_ROOT}"
  echo "[STAGE] seed=${SEED} training starts $(date -Is)"
  python -m semantic_constraints.cst_teacher.train_teacher \
    --device cuda \
    --fold 0 \
    --seed "${SEED}" \
    --epochs 80 \
    --patience 12 \
    --minimum-improvement 1e-4 \
    --batch-size 32 \
    --num-workers 0 \
    --channels 16 32 64 \
    --heads 4 \
    --set-size 12 \
    --minimum-set-size 6 \
    --slab-depth 3 \
    --inplane-size 32 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --output-dir "${SEED_ROOT}" \
    2>&1 | tee "${SEED_ROOT}/train.log"

  test -f "${SEED_ROOT}/best_teacher.pt"
  echo "[STAGE] seed=${SEED} embedding analysis starts $(date -Is)"
  python -m semantic_constraints.cst_teacher.analyze_embeddings \
    --checkpoint "${SEED_ROOT}/best_teacher.pt" \
    --device cuda \
    --batch-size 32 \
    --max-clusters 4 \
    --minimum-cluster-size 10 \
    --bootstrap-samples 250 \
    --seed "${SEED}" \
    --output-dir "${SEED_ROOT}/embedding_analysis" \
    > "${SEED_ROOT}/embedding_analysis.log"

  echo "[STAGE] seed=${SEED} prediction clue evaluation starts $(date -Is)"
  python -m semantic_constraints.cst_teacher.evaluate_predictions \
    --checkpoint "${SEED_ROOT}/best_teacher.pt" \
    --inference-dir "${INFERENCE_DIR}" \
    --device cuda \
    --batch-size 32 \
    --output "${SEED_ROOT}/prediction_evaluation.json" \
    > "${SEED_ROOT}/prediction_evaluation.log"
done

python -m semantic_constraints.cst_teacher.summarize_study \
  --run-root "${RUN_ROOT}" \
  --seeds "${SEEDS[@]}" \
  > "${RUN_ROOT}/study_summary.log"

echo "[DONE] $(date -Is)"
echo "[DONE] summary=${RUN_ROOT}/study_summary.md"
cat "${RUN_ROOT}/study_summary.md"
