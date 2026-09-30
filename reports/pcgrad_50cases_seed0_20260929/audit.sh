#!/usr/bin/env bash
#SBATCH --job-name=pcgrad50-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
AUDIT=/mnt/beegfsstudents/home/3160552/pcgrad_50cases_audit_20260929_01
cd "$AUDIT"
/home/3160552/.conda/envs/hippocampus/bin/python recompute_audit.py --home /mnt/beegfsstudents/home/3160552 --output "$AUDIT/results"
