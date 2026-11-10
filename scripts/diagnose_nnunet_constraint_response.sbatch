#!/usr/bin/env bash
#SBATCH --job-name=hippo-nn-constraint-diagnosis
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/nnunet_constraint_diagnosis_${SLURM_JOB_ID:?}"
parent="/mnt/beegfsstudents/home/3160552/four_models_six_optimization_31976_685853"
mkdir -p "${root}"
if [[ ! -d "${root}/source" ]]; then
  cp -r "${parent}/source" "${root}/source"
  cp "${parent}/source_manifest.json" "${root}/source_manifest.json"
  cp "${stage}/diagnose_nnunet_constraint_response.py" "${root}/source/scripts/"
fi
cmp "${stage}/diagnose_nnunet_constraint_response.py" "${root}/source/scripts/diagnose_nnunet_constraint_response.py"
export PYTHONPATH="${root}/source:${root}/source/scripts:${root}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-nn-diagnosis-${SLURM_JOB_ID}"
/home/3160552/.conda/envs/hippocampus/bin/python "${root}/source/scripts/diagnose_nnunet_constraint_response.py" \
  --root "${root}" --parent-root "${parent}" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
