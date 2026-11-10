#!/usr/bin/env bash
#SBATCH --job-name=unet48-stable-budget
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-stability-${SLURM_JOB_ID}"
root="/mnt/beegfsstudents/home/3160552/unet48_stable_budget_${SLURM_JOB_ID}"
export PYTHONPATH="${stage}/source:${stage}/source/scripts"
/home/3160552/.conda/envs/hippocampus/bin/python "${stage}/source/scripts/unet48_stability/verify_replication.py" --output "${stage}/replication_qualification.json"
mkdir -p "${root}"
cp -r "${stage}/source" "${root}/source"
cp "${stage}/source_manifest.json" "${root}/runtime_origin_manifest.json"
/home/3160552/.conda/envs/hippocampus/bin/python "${root}/source/scripts/run_unet48_budget_selection.py" \
 --root "${root}" --plan "${root}/source/experiments/unet48_stable_budget/plan.json" \
 --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl --threads 4
