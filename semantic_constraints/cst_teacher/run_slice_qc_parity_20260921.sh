#!/usr/bin/env bash
# Verify that label-free feature extraction matches archived CST risk features.
#SBATCH --job-name=cst-qc-parity
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
trap 'rc=$?; printf "[ERROR] exit=%s line=%s command=%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2; exit "$rc"' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_BASE="${CST_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${CST_RUN_ROOT:-${RUN_BASE}/cst_qc_parity_${SLURM_JOB_ID:-local}}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CST_ENV_NAME:-hippocampus}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QC_PARITY_RUN_ROOT="${RUN_ROOT}"
export QC_PARITY_RUN_BASE="${RUN_BASE}"

python - <<'PY'
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from semantic_constraints.cst_teacher.evaluate_predictions import load_teachers, prediction_map
from semantic_constraints.cst_teacher.extract_slice_qc_features import extract_arrays
from semantic_constraints.cst_teacher.slice_qc import load_ridge, ridge_predict
from semantic_constraints.cst_teacher.train_teacher import build_loaders

repo = Path.cwd()
base = Path(os.environ["QC_PARITY_RUN_BASE"])
run = Path(os.environ["QC_PARITY_RUN_ROOT"])
root = base / "cst_early_stopped_reanalysis_665422/fold0"
teacher_path = base / "cst_profile_ablation_665313/dense32_smooth_seed_0/best_teacher.pt"
teacher, _, payload = load_teachers(teacher_path, torch.device("cuda"))
config = payload["config"]
args = argparse.Namespace(
    pkl=repo / "datasets/Dataset101_MSD/msd_hippocampus_full.pkl",
    splits_json=repo / "datasets/Dataset101_MSD/splits_final.json",
    fold=0,
    spatial_size=tuple(config["spatial_size"]),
    set_size=config["set_size"],
    slab_depth=config["slab_depth"],
    inplane_size=config["inplane_size"],
    cache_dataset=False,
    max_train_cases=0,
    max_val_cases=0,
    batch_size=16,
    num_workers=0,
)
_, loader = build_loaders(args)
archived = np.load(root / "risk/risk_features_seed_0.npz", allow_pickle=False)
paths = prediction_map(root / "inference")
head_path = base / "cst_slice_qc_665471/seed_0/models/fold0/ridge_combined.npz"
kind, head, _ = load_ridge(head_path)
assert kind == "combined"
records = []
for index in range(3):
    item = loader.dataset.base_dataset[index]
    name = str(item["case_name"])
    probabilities = np.load(paths[name], allow_pickle=False)
    if index == 0:
        np.save(run / "preprocessed_case.npy", np.asarray(item["image"], dtype=np.float32))
        np.save(run / "probabilities_case.npy", probabilities)
    current = extract_arrays(
        teacher,
        np.asarray(item["image"]),
        probabilities,
        set_size=config["set_size"],
        slab_depth=config["slab_depth"],
        inplane_size=config["inplane_size"],
        device="cuda",
    )
    reference_indices = np.flatnonzero(archived["slice_groups"] == name)
    if len(reference_indices) != config["set_size"]:
        raise ValueError(f"archived feature count mismatch for {name}")
    differences = {}
    for key in ("slice_uncertainty", "slice_relationship_residuals", "slice_combined"):
        expected = archived[key][reference_indices]
        drift = np.abs(current[key] - expected)
        differences[key] = {
            "maximum_absolute_difference": float(drift.max()),
            "mean_absolute_difference": float(drift.mean()),
            "fraction_above_0.001": float(np.mean(drift > 0.001)),
        }
    current_score = ridge_predict(head, current["slice_combined"][None])[0]
    archived_score = ridge_predict(head, archived["slice_combined"][reference_indices][None])[0]
    score_drift = np.abs(current_score - archived_score)
    records.append({
        "case_name": name,
        "feature_differences": differences,
        "maximum_score_difference": float(score_drift.max()),
        "mean_score_difference": float(score_drift.mean()),
    })
    print(records[-1], flush=True)

feature_maximum = max(
    details["maximum_absolute_difference"]
    for record in records for details in record["feature_differences"].values()
)
score_maximum = max(record["maximum_score_difference"] for record in records)
report = {
    "schema": "semantic_constraints.cst_teacher.slice_qc_feature_parity.v1",
    "device": "cuda",
    "cases": records,
    "maximum_feature_difference": feature_maximum,
    "maximum_score_difference": score_maximum,
    # Archived features used 16-case teacher batches; this CLI uses one case.
    # Kernel-order roundoff is acceptable only when the deployed score remains
    # substantially closer than the feature values themselves.
    "passed": feature_maximum < 0.002 and score_maximum < 0.001,
}
(run / "parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise AssertionError("single-case versus archived-batch QC parity exceeds tolerance")
PY

python -m semantic_constraints.cst_teacher.extract_slice_qc_features \
  --teacher-checkpoint "${RUN_BASE}/cst_profile_ablation_665313/dense32_smooth_seed_0/best_teacher.pt" \
  --preprocessed-image-npy "${RUN_ROOT}/preprocessed_case.npy" \
  --swin-probabilities-npy "${RUN_ROOT}/probabilities_case.npy" \
  --case-name hippocampus_017 \
  --output "${RUN_ROOT}/label_free_features.npz" --device cuda
python -m semantic_constraints.cst_teacher.score_slice_qc \
  --model "${RUN_BASE}/cst_slice_qc_665471/seed_0/models/fold1/ridge_portable.npz" \
  --features "${RUN_ROOT}/label_free_features.npz" \
  --output "${RUN_ROOT}/label_free_scores.csv"
python - <<'PY'
import csv
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["QC_PARITY_RUN_ROOT"])
with np.load(root / "label_free_features.npz", allow_pickle=False) as features:
    assert "slice_target" not in features.files
with (root / "label_free_scores.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 32
assert {row["case_name"] for row in rows} == {"hippocampus_017"}
print("Label-free extraction and scoring CLI passed for 32 slices", flush=True)
PY

echo "[DONE] $(date -Is)"
