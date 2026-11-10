#!/usr/bin/env bash
#SBATCH --job-name=hippo-edge-recovery
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
set -Eeuo pipefail
run_root="${1:?Supply original campaign root}"
repair_root="${2:?Supply frozen recovery root}"
export PYTHONPATH="${run_root}/source"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-edge-recovery-${SLURM_JOB_ID}"
cd "${run_root}/source"
exec /home/3160552/.conda/envs/hippocampus/bin/python -u "${repair_root}/recover.py" worker --root "${run_root}" --repair "${repair_root}"
