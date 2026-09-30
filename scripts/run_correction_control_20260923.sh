#!/bin/bash
#SBATCH --job-name=correction-control
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=correction-control-%j.out
#SBATCH --error=correction-control-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
sha256sum --check MANIFEST.sha256
correction_args=(
  --source "$HOME/function_head_robustness_20260922"
  --root "$HOME/hippopotamus_runs/boundary_function_robustness_20260922_666174"
  --pkl "$HOME/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl"
  --splits "$HOME/function_head_robustness_20260922/datasets/Dataset101_MSD/splits_final.json"
  --reference "$HOME/presence_decisive_20260923_v2/results_audit_666651/provenance.json"
  --output "$SLURM_SUBMIT_DIR/results_${SLURM_JOB_ID}"
)
"$HOME/.conda/envs/hippocampus/bin/python" -u train_correction_control.py "${correction_args[@]}" --stage fit
"$HOME/.conda/envs/hippocampus/bin/python" -u train_correction_control.py "${correction_args[@]}" --stage evaluate
