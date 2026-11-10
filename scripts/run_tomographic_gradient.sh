#!/usr/bin/env bash
#SBATCH --job-name=tomographic-gradient
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run_root="${1:?Supply new frozen experiment directory}"
export PYTHONPATH="${run_root}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "${run_root}/source"
python_bin=/home/3160552/.conda/envs/hippocampus/bin/python
"${python_bin}" -m unittest semantic_constraints.tomographic.test_losses semantic_constraints.tomographic.test_probe -v
"${python_bin}" -m semantic_constraints.tomographic.probe \
  --pkl /home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
  --splits /home/3160552/hippo/datasets/Dataset101_MSD/splits_final.json \
  --checkpoint /home/3160552/budget_baselines_94603_684354/swinunetr/fold0/seed0/run/checkpoint_best.pt \
  --cases 32 --synthetic-cases 8 --output "${run_root}/results"
