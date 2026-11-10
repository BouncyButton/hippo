#!/usr/bin/env bash
#SBATCH --job-name=geofwi3d-3fold-3seed
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
CAMPAIGN="$1"
GEOFWI3D_PYTHON="$2"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${CAMPAIGN}/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${CAMPAIGN}"
"${GEOFWI3D_PYTHON}" source/geofwi3d_replication.py run --campaign "${CAMPAIGN}"
