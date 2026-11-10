#!/usr/bin/env bash
# Submit one job from the cluster repository root:
#   sbatch scripts/run_compute_matched_baselines.sbatch
# All 27 model/fold/seed experiments run sequentially on one allocated GPU.
#SBATCH --job-name=hippo-compute-baselines
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${HIPPO_PYTHON:-/home/3160552/.conda/envs/hippocampus/bin/python}"
data_pkl="${HIPPO_DATA_PKL:-/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl}"
job_id="${SLURM_JOB_ID:?Submit this file with sbatch as a single job}"
run_root="${HIPPO_COMPUTE_RUN_ROOT:-/mnt/beegfsstudents/home/3160552/compute_matched_baselines_${job_id}}"
models=(unet swinunetr nnunet)
source_root="${run_root}/source"

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-compute-mpl-${job_id}"
export nnUNet_compile=false

test -x "${python_bin}"
test -f "${data_pkl}"
test -f "${repo_root}/experiments/compute_matched_baselines_20261003/protocol.json"
mkdir -p "${run_root}"

# All 27 experiments share one immutable campaign snapshot. A setup lock also
# protects publication if this campaign is resumed in a later job.
exec 9>"${run_root}/source.lock"
flock 9
if [[ ! -d "${source_root}" ]]; then
  stage_root="$(mktemp -d "${run_root}/source-stage-XXXXXX")"
  tar -C "${repo_root}" --exclude='__pycache__' --exclude='*.pyc' -cf - \
    scripts/train_compute_matched_baselines.py \
    scripts/summarize_compute_matched_baselines.py \
    scripts/run_compute_matched_baselines.sbatch \
    experiments/compute_matched_baselines_20261003 \
    baselines/swin_unetr/swin_unetr.py \
    baselines/nnUNet/nnunetv2 \
    thesis/new_constraints/training_augmentation.py \
    | tar -C "${stage_root}" -xf -
  mv "${stage_root}" "${source_root}"
fi
export PYTHONPATH="${source_root}:${source_root}/baselines/nnUNet${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" - "${source_root}" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import monai
import torch

root = Path(sys.argv[1])
if monai.__version__ != '1.6.0':
    raise RuntimeError('Use the profiled MONAI 1.6.0 environment')
version = importlib.metadata.version('dynamic-network-architectures')
parts = tuple(int(x) for x in version.split('.')[:3])
if not (0, 4, 1) <= parts < (0, 5, 0):
    raise RuntimeError('dynamic-network-architectures 0.4.x, at least 0.4.1, required')
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('An allocated CUDA GPU with bfloat16 support is required')
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob('*') if p.is_file()}
manifest = root.parent / 'source_manifest.json'
if manifest.exists() and json.loads(manifest.read_text()) != hashes:
    raise RuntimeError('Frozen campaign source changed')
manifest.write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({'torch': torch.__version__, 'monai': monai.__version__,
                 'dynamic_network_architectures': version,
                 'gpu': torch.cuda.get_device_name(0)}, indent=2))
PY
flock -u 9

for model in "${models[@]}"; do
  for fold in 0 1 2; do
    for seed in 0 1 2; do
      task_root="${run_root}/${model}/fold${fold}/seed${seed}"
      mkdir -p "${task_root}"
      echo "Training ${model}, fold ${fold}, seed ${seed}; results: ${task_root}/run"
      "${python_bin}" "${source_root}/scripts/train_compute_matched_baselines.py" \
        --model "${model}" --fold "${fold}" --seed "${seed}" \
        --pkl "${data_pkl}" --output "${task_root}/run" \
        --protocol-dir "${source_root}/experiments/compute_matched_baselines_20261003" \
        --threads "${SLURM_CPUS_PER_TASK:-4}" --resume
    done
  done
done

# Publish the verified aggregate after all 27 experiments complete.
"${python_bin}" "${source_root}/scripts/summarize_compute_matched_baselines.py" \
  --root "${run_root}"
