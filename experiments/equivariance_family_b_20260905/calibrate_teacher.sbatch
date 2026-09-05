#!/usr/bin/env bash
#SBATCH --job-name=teacher-train-calibration
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus
export PYTHONDONTWRITEBYTECODE=1
python -m thesis.new_constraints.teacher.calibrate_training \
  --run-dir /mnt/beegfsstudents/home/3160552/onecut_protocol_20260902/msd_fold0_none_calibration_seed0 \
  --max-cases 32 --output "${1:?new report path required}"
