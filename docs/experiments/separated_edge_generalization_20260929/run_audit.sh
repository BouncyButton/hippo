#!/usr/bin/env bash
#SBATCH --job-name=edge-generalization-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
set -euo pipefail
ROOT="${1:?Pass audit output root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
/home/3160552/.conda/envs/hippocampus/bin/python "$ROOT/recompute_bands_cuda.py" --experiment /mnt/beegfsstudents/home/3160552/separated_edge_20260929_01 --output "$ROOT"
