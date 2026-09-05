#!/usr/bin/env bash
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail
ROOT="/mnt/beegfsstudents/home/3160552/equivariance_family_b_20260905_01"
SEED="${1:?seed required}"
PREVIOUS="${2:-}"
[[ "$SEED" == 0 || "$SEED" == 1 ]]
OUTPUT="$ROOT/msd_fold0_equivariance_seed${SEED}"
RESUME=()
if [[ -n "$PREVIOUS" ]]; then
  if [[ -f "$OUTPUT/completion_manifest.json" ]]; then
    echo "Run already completed; no continuation needed."
    exit 0
  fi
  STATE="$(sacct -X -j "$PREVIOUS" -n -o State | xargs)"
  case "$STATE" in
    TIMEOUT*|PREEMPTED*|NODE_FAIL*) ;;
    *) echo "Refusing automatic retry after $STATE; inspect job $PREVIOUS." >&2; exit 1 ;;
  esac
  [[ -f "$OUTPUT/checkpoint_latest.pt" ]]
  RESUME=(--resume)
fi
cd "$ROOT/source"
export PYTHONDONTWRITEBYTECODE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus
bash thesis/new_constraints/run_new_constraints_cluster.sh \
  --constraint-set equivariance --equivariance-weight 0.1 \
  --translation-size 2 --equivariance-max-samples 1 \
  --dataset MSD --fold 0 --seed "$SEED" --epochs 50 --batch-size 1 \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --constraint-warmup-epochs 5 --constraint-eval-every 0 \
  --output-dir "$OUTPUT" "${RESUME[@]}"
