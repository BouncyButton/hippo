#!/usr/bin/env bash
#SBATCH --job-name=spectral-shape-f0
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
run_root="${1:?Supply the experiment directory}"
export PYTHONPATH="${run_root}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/spectral-shape-mpl-${SLURM_JOB_ID}"
cd "${run_root}/source"
python_bin=/home/3160552/.conda/envs/hippocampus/bin/python
"${python_bin}" -m unittest semantic_constraints.spectral_shape.test_audit
"${python_bin}" -m semantic_constraints.spectral_shape.audit \
  --checkpoint "${run_root}/baseline_seed0_checkpoint_best.pt" \
  --pkl /home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits /home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --output "${run_root}/results"
