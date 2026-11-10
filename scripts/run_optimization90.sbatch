#!/usr/bin/env bash
# Twenty-four fresh runs sequentially; calibration reported separately.
#SBATCH --job-name=hippo-opt-only-31976
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/four_models_six_optimization_31976_${SLURM_JOB_ID:?}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-optimization-${SLURM_JOB_ID}"
export nnUNet_compile=false
mkdir -p "${root}"
if [[ ! -d "${root}/source" ]]; then
  cp -r "${stage}/source" "${root}/source"
  cp "${stage}/source_manifest.json" "${root}/source_manifest.json"
  cp "${stage}/preflight.json" "${root}/preflight.json"
fi
cmp "${stage}/source_manifest.json" "${root}/source_manifest.json"
export PYTHONPATH="${root}/source:${root}/source/scripts:${root}/source/baselines/nnUNet"
python_bin="/home/3160552/.conda/envs/hippocampus/bin/python"
"${python_bin}" "${root}/source/scripts/run_optimization90.py" --root "${root}" \
  --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl
