#!/usr/bin/env bash
#SBATCH --job-name=edge-coherence
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
AUDIT_ROOT=/mnt/beegfsstudents/home/3160552/edge_coherence_20260928_01
cd "$AUDIT_ROOT"
/home/3160552/.conda/envs/hippocampus/bin/python audit_edge_coherence.py \
  --base /mnt/beegfsstudents/home/3160552 --output "$AUDIT_ROOT/results"
