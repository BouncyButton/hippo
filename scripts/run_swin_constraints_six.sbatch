#!/usr/bin/env bash
# One allocation: six SwinUNETR objectives, fold 0, seed 0.
#SBATCH --job-name=swin-six-bref12207-f0-s0
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
trap 'status=$?; printf "FAILED status=%s line=%s command=%s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR
stage="${SLURM_SUBMIT_DIR:?}"
root="/mnt/beegfsstudents/home/3160552/compute_matched_swin_six_${SLURM_JOB_ID:?}"
python_bin="/home/3160552/.conda/envs/hippocampus/bin/python"
source_root="${root}/source"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-swin-six-${SLURM_JOB_ID}"
mkdir -p "${root}"
exec 9>"${root}/source.lock"
flock -n 9
if [[ ! -d "${source_root}" ]]; then
  source_stage="$(mktemp -d "${root}/source-stage-XXXXXX")"
  tar -C "${stage}/frozen_runtime" --exclude='__pycache__' --exclude='*.pyc' -cf - . | tar -C "${source_stage}" -xf -
  mv "${source_stage}" "${source_root}"
fi
export PYTHONPATH="${source_root}:${source_root}/scripts"
protocol_dir="${source_root}/experiments/compute_matched_swin_six_20261004"
"${python_bin}" - "${source_root}" "${stage}" "${root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import monai,torch
source,stage,root=map(Path,sys.argv[1:])
assert monai.__version__=='1.6.0' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
expected=json.loads((stage/'runtime_origin_manifest.json').read_text())
for name,sha in expected.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
p=json.loads((source/'experiments/compute_matched_swin_six_20261004/protocol.json').read_text())
assert p['budget_flops']==12207124453303 and p['folds']==p['seeds']==[0]
assert list(p['models'])==['swinunetr']
assert list(p['models']['swinunetr']['objectives'])==['dice','dice_bands','dice_sdf','dice_bands_sdf','dice_bands_edge','dice_bands_edge_sdf']
assert hashlib.sha256((source/'experiments/compute_budget_reference.json').read_bytes()).hexdigest()==p['reference_sha256']
(root/'source_manifest.json').write_text(json.dumps(expected,indent=2)+'\n')
print(json.dumps(dict(torch=torch.__version__,monai=monai.__version__,gpu=torch.cuda.get_device_name(0))))
PY
for objective in dice dice_bands dice_sdf dice_bands_sdf dice_bands_edge dice_bands_edge_sdf; do
  printf 'START objective=%s fold=0 seed=0\n' "${objective}"
  "${python_bin}" "${source_root}/scripts/train_compute_matched_constraints.py" \
    --model swinunetr --objective "${objective}" --fold 0 --seed 0 \
    --pkl /mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl \
    --output "${root}/swinunetr/${objective}/fold0/seed0/run" \
    --protocol-dir "${protocol_dir}" --threads "${SLURM_CPUS_PER_TASK:-4}" --resume
done
"${python_bin}" - "${root}" "${protocol_dir}" <<'PY'
from pathlib import Path
import sys
import summarize_compute_matched_constraints as reporter
reporter.DEFAULT_PROTOCOL=Path(sys.argv[2])
reporter.summarize(Path(sys.argv[1]))
PY
