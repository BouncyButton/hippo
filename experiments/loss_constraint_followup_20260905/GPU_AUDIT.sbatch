#!/usr/bin/env bash
# Submit from an isolated audit source snapshot. This job performs no training.
#SBATCH --job-name=hippo-official-audit
#SBATCH --partition=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hippocampus
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python -m thesis.new_constraints.audit_official_cluster "$@"
