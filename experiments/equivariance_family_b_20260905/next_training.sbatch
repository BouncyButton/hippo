#!/usr/bin/env bash
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail
# Run from a fresh, frozen source snapshot.
MODE="${1:?augmentation, augmentation_equivariance, or teacher}"
SEED="${2:?seed required}"
OUTPUT="${3:?new absolute output directory required}"
[[ "$OUTPUT" == /* && ( "$SEED" == 0 || "$SEED" == 1 ) ]]
[[ -f thesis/new_constraints/teacher/calibrate_training.py ]]
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus
export PYTHONDONTWRITEBYTECODE=1
FLAGS=(--constraint-set none)
case "$MODE" in
  augmentation) FLAGS+=(--translation-augmentation) ;;
  augmentation_equivariance)
    FLAGS=(--constraint-set equivariance --equivariance-weight 0.1 --translation-augmentation) ;;
  teacher)
    REPORT="${4:?training-only calibration report required}"
    WEIGHT="$(python - "$REPORT" <<'PY'
import json, math, sys
from thesis.new_constraints.source_bootstrap import source_digest
from thesis.new_constraints.train_swinunetr_constraints import collect_runtime_provenance
r = json.load(open(sys.argv[1]))
assert r['status'] == 'complete' and r['purpose'] == 'training_only_logit_scale_calibration'
assert r['source_sha256'] == source_digest() and r['runtime'] == collect_runtime_provenance()
assert r['cohort'] == 'fold0_train_only' and len(r['cases']) == 32
assert r['supervised_loss'] == 'dice' and not r['translation_augmentation']
assert r['teacher']['support'] == 'common' and r['teacher']['views'] == 2
assert r['teacher']['temperature'] == 1
w = r['recommended_weight']
assert math.isfinite(w) and w > 0
print(w)
PY
)"
    FLAGS=(--constraint-set teacher --teacher-weight "$WEIGHT" --teacher-views 2
           --teacher-temperature 1 --teacher-support common) ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
RESUME=(); [[ "${RESUME_RUN:-0}" != 1 ]] || RESUME=(--resume)
bash thesis/new_constraints/run_new_constraints_cluster.sh \
  "${FLAGS[@]}" --supervised-loss dice --fold 0 --seed "$SEED" \
  --epochs 50 --batch-size 1 --constraint-eval-every 0 --constraint-warmup-epochs 5 \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits-json /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --output-dir "$OUTPUT" "${RESUME[@]}"
