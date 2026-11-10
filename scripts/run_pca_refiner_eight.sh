#!/usr/bin/env bash
#SBATCH --job-name=pca-refiner-eight
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
run_root="${1:?Supply run directory}"
export PYTHONPATH="${run_root}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pca-refiner-eight-${SLURM_JOB_ID}"
cd "${run_root}/source"
python_bin=/home/3160552/.conda/envs/hippocampus/bin/python
"${python_bin}" -m unittest semantic_constraints.spectral_shape.test_audit semantic_constraints.pca_refiner.test_fit_eight
"${python_bin}" -m semantic_constraints.pca_refiner.fit_eight \
  --checkpoint /home/3160552/spectral_shape_fold0_20260930_01/baseline_seed0_checkpoint_best.pt \
  --pkl /home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits /home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --output "${run_root}/results"
"${python_bin}" -m semantic_constraints.pca_refiner.verify "${run_root}/results"
