#!/usr/bin/env bash
#SBATCH --job-name=step-head-gate-fold1
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=step-head-gate-fold1-%j.out
#SBATCH --error=step-head-gate-fold1-%j.err

set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
RUN_BASE="${STEP_GATE_RUN_BASE:-$(dirname "${REPO_ROOT}")/hippopotamus_runs}"
RUN_ROOT="${STEP_GATE_RUN_ROOT:-${RUN_BASE}/step_head_gate_fold1_20260922_${SLURM_JOB_ID}}"
mkdir -p "${RUN_ROOT}/source"
SOURCE="${RUN_ROOT}/source/step_head_gate_followup.py"
if [[ ! -f "${SOURCE}" ]]; then
  cp "${REPO_ROOT}/thesis/new_constraints/step_head_gate_followup.py" "${SOURCE}"
  cp "${REPO_ROOT}/thesis/new_constraints/step_head_pilot.py" "${RUN_ROOT}/source/step_head_pilot.py"
fi
PYTHON_BIN="${STEP_GATE_PYTHON:-${HOME}/.conda/envs/hippocampus/bin/python}"
test -x "${PYTHON_BIN}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"${PYTHON_BIN}" "${SOURCE}" \
  --pkl "${REPO_ROOT}/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
  --splits "${REPO_ROOT}/datasets/Dataset101_MSD/splits_final.json" \
  --output "${RUN_ROOT}" \
  2>&1 | tee -a "${RUN_ROOT}/followup.log"
