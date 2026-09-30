#!/usr/bin/env bash
#SBATCH --job-name=boundary-causal-pilot
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
ROOT="${1:?experiment root}"
export PYTHONPATH="$ROOT/source:$ROOT/source/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
cd "$ROOT/source"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01/test_runtime:$PYTHONPATH" "$PYTHON" -m pytest -q scripts/test_boundary_causal.py > "$ROOT/GPU_TESTS.txt" 2>&1
"$PYTHON" scripts/train_boundary_causal.py --root "$ROOT" --seed 0 --arms sum dice sum_aug dice_aug
