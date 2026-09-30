#!/usr/bin/env bash
#SBATCH --job-name=boundary-diversity
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
ROOT="${1:?root}"
export PYTHONPATH="$ROOT/source:$ROOT/source/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$ROOT/source"
/home/3160552/.conda/envs/hippocampus/bin/python scripts/train_boundary_diversity.py --root "$ROOT" --reference /mnt/beegfsstudents/home/3160552/boundary_causal_20260929_01/seed0/sum
