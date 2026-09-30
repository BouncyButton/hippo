#!/usr/bin/env bash
# Export training-set probabilities from the archived stopped nnU-Net checkpoint.
# Submit with: sbatch scripts/run_nnunet_continuity_probabilities.sh
#SBATCH --job-name=nnunet-continuity-probs
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -Eeuo pipefail

RUN_ROOT=/mnt/beegfsstudents/home/3160552/nnunet_recovery_20260920_01/run
AUDIT_ROOT=/mnt/beegfsstudents/home/3160552/nnunet_recovery_20260920_01/audit_stopped_checkpoint
MODEL_BASE="${RUN_ROOT}/results/Dataset101_MSD/nnUNetTrainer_50epochsEarlyStopping__nnUNetPlans__3d_fullres"
OUTPUT_ROOT=/mnt/beegfsstudents/home/3160552/nnunet_recovery_20260920_01/continuity_probabilities_20260922

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw="${RUN_ROOT}/datasets"
export nnUNet_preprocessed="${RUN_ROOT}/preprocessed"
export nnUNet_results="${RUN_ROOT}/results"

test -f "${MODEL_BASE}/fold_0/checkpoint_final.pth"
test -d "${AUDIT_ROOT}/inputs/train"
mkdir -p "${OUTPUT_ROOT}/train"

nnUNetv2_predict_from_modelfolder \
  -i "${AUDIT_ROOT}/inputs/train" \
  -o "${OUTPUT_ROOT}/train" \
  -m "${MODEL_BASE}" -f 0 -chk checkpoint_final.pth \
  -npp 2 -nps 2 --save_probabilities --continue_prediction \
  --disable_progress_bar

python - "${OUTPUT_ROOT}/train" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
probabilities = list(root.glob("hippocampus_*.npz"))
if len(probabilities) != 208:
    raise RuntimeError(f"Expected 208 probability files, found {len(probabilities)}")
print(f"Exported {len(probabilities)} probability files to {root}")
PY
