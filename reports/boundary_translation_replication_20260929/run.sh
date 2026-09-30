#!/usr/bin/env bash
#SBATCH --job-name=boundary-tta-replicate
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
ROOT="${1:?root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/3160552/.conda/envs/hippocampus/bin/python "$ROOT/audit.py" --pilot /mnt/beegfsstudents/home/3160552/boundary_replication_20260929_01 --seed 1 --arms dice_aug dice --out "$ROOT/results"
