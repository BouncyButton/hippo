#!/bin/bash
#SBATCH --job-name=correction-discrimination
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=discrimination-%j.out
#SBATCH --error=discrimination-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
sha256sum --check MANIFEST.sha256
export OMP_NUM_THREADS=8
export PYTHONPATH="$HOME/correction_control_20260923_01a0cd${PYTHONPATH:+:$PYTHONPATH}"
"$HOME/.conda/envs/hippocampus/bin/python" -u audit_correction_discrimination.py \
  --source "$HOME/function_head_robustness_20260922" \
  --root "$HOME/hippopotamus_runs/boundary_function_robustness_20260922_666174" \
  --pkl "$HOME/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl" \
  --splits "$HOME/function_head_robustness_20260922/datasets/Dataset101_MSD/splits_final.json" \
  --reference "$HOME/presence_decisive_20260923_v2/results_audit_666651/provenance.json" \
  --saved "$HOME/correction_control_20260923_01a0cd/results_666769" \
  --output "$SLURM_SUBMIT_DIR/results_${SLURM_JOB_ID}"
