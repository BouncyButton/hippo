#!/usr/bin/env bash
# One fresh nnU-Net SDF-only timing pilot, fold0/seed0.
#SBATCH --job-name=hippo-nn-delayed-sdf-only
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/nnunet_delayed_sdf_only_31976_${SLURM_JOB_ID:?}"
mkdir -p "${root}"
if [[ ! -d "${root}/source" ]]; then
  cp -r "${stage}/source" "${root}/source"
  cp "${stage}/source_manifest.json" "${root}/source_manifest.json"
  cp "${stage}/preflight.json" "${root}/preflight.json"
fi
cmp "${stage}/source_manifest.json" "${root}/source_manifest.json"
export PYTHONPATH="${root}/source:${root}/source/scripts:${root}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-delayed-sdf-${SLURM_JOB_ID}"
/home/3160552/.conda/envs/hippocampus/bin/python "${root}/source/scripts/delayed_sdf_only_pilot.py" run \
  --root "${root}" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
