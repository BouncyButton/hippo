#!/usr/bin/env bash
# One allocation: baseline-only width-48 budget selection; no constraint runs.
#SBATCH --job-name=unet48-budget70-75-f0-s0
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
trap 'status=$?; printf "FAILED status=%s line=%s command=%s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/compute_budget_unet48_${SLURM_JOB_ID:?}"
python_bin="/home/3160552/.conda/envs/hippocampus/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-unet48-budget-${SLURM_JOB_ID}"
mkdir -p "${root}"
exec 9>"${root}/controller.lock"
flock -n 9
if [[ ! -d "${root}/source" ]]; then
  source_stage="$(mktemp -d "${root}/source-stage-XXXXXX")"
  tar -C "${stage}/frozen_runtime" --exclude='__pycache__' --exclude='*.pyc' -cf - . | tar -C "${source_stage}" -xf -
  mv "${source_stage}" "${root}/source"
  cp "${stage}/runtime_origin_manifest.json" "${root}/runtime_origin_manifest.json"
fi
export PYTHONPATH="${root}/source:${root}/source/scripts"
"${python_bin}" "${root}/source/scripts/run_unet48_budget_selection.py" \
  --root "${root}" --plan "${root}/source/experiments/unet48_budget_selection_20261005/plan.json" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
