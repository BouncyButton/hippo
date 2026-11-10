#!/usr/bin/env bash
# Conditional single bands comparison, paired with baseline684625.
#SBATCH --job-name=hippo-unet16-bands
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/plain_unet16_bands_${SLURM_JOB_ID:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-unet16-bands-${SLURM_JOB_ID}"
mkdir -p "${root}"
if [[ ! -d "${root}/source" ]]; then
  cp -r "${stage}/source" "${root}/source"
  cp "${stage}/source_manifest.json" "${root}/source_manifest.json"
fi
cmp "${stage}/source_manifest.json" "${root}/source_manifest.json"
export PYTHONPATH="${root}/source:${root}/source/scripts"
/home/3160552/.conda/envs/hippocampus/bin/python "${root}/source/scripts/run_plain_unet16_bands.py" \
  --root "${root}" --unetr-root "${UNETR_REFERENCE_ROOT:?}" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
