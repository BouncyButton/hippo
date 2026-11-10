#!/usr/bin/env bash
# One allocation, no array: U-Net x eight objectives x fold 0 x seed 0.
#SBATCH --job-name=unet-bref20-sdf-f0-s0
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -Eeuo pipefail
trap 'status=$?; printf "FAILED status=%s line=%s command=%s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR
repo_root="${SLURM_SUBMIT_DIR:?Submit from the staged repository root}"
python_bin="${HIPPO_PYTHON:-/home/3160552/.conda/envs/hippocampus/bin/python}"
data_pkl="${HIPPO_DATA_PKL:-/mnt/beegfsstudents/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl}"
job_id="${SLURM_JOB_ID:?Submit as one sbatch job}"
run_root="${HIPPO_CONSTRAINT_RUN_ROOT:-/mnt/beegfsstudents/home/3160552/compute_matched_constraints20_${job_id}}"
source_root="${run_root}/source"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hippo-constraints-mpl-${job_id}"
mkdir -p "${run_root}"
exec 9>"${run_root}/source.lock"
flock 9
if [[ ! -d "${source_root}" ]]; then
  stage_root="$(mktemp -d "${run_root}/source-stage-XXXXXX")"
  tar -C "${repo_root}" --exclude='__pycache__' --exclude='*.pyc' -cf - \
    scripts/train_compute_matched_baselines.py \
    scripts/compute_constraint_objectives.py \
    scripts/train_compute_matched_constraints.py \
    scripts/summarize_compute_matched_constraints.py \
    scripts/run_compute_matched_constraints.sbatch \
    experiments/compute_matched_constraints_20pct_20261004 \
    experiments/compute_budget_reference.json \
    baselines/swin_unetr/swin_unetr.py \
    thesis/new_constraints \
    | tar -C "${stage_root}" -xf -
  mv "${stage_root}" "${source_root}"
fi
export PYTHONPATH="${source_root}:${source_root}/scripts"
"${python_bin}" - "${source_root}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import monai,torch
root=Path(sys.argv[1])
assert monai.__version__=='1.6.0' and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
p=json.loads((root/'experiments/compute_matched_constraints_20pct_20261004/protocol.json').read_text())
assert p['status']=='ready' and p['folds']==p['seeds']==[0] and list(p['models'])==['unet']
assert p['budget_flops']==8138082968869 and len(p['models']['unet']['objectives'])==8
assert hashlib.sha256((root/'experiments/compute_budget_reference.json').read_bytes()).hexdigest()==p['reference_sha256']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
manifest=root.parent/'source_manifest.json'
if manifest.exists() and json.loads(manifest.read_text())!=hashes:
    raise RuntimeError('Frozen source changed')
manifest.write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(dict(torch=torch.__version__,monai=monai.__version__,gpu=torch.cuda.get_device_name(0))))
PY
flock -u 9
for model in unet; do
  for objective in dice dice_bands dice_bands_edge dice_edge dice_sdf dice_bands_sdf dice_bands_edge_sdf dice_edge_sdf; do
    for fold in 0; do
      for seed in 0; do
        task_root="${run_root}/${model}/${objective}/fold${fold}/seed${seed}/run"
        printf 'START model=%s objective=%s fold=%s seed=%s\n' "$model" "$objective" "$fold" "$seed"
        "${python_bin}" "${source_root}/scripts/train_compute_matched_constraints.py" \
          --model "${model}" --objective "${objective}" --fold "${fold}" --seed "${seed}" \
          --pkl "${data_pkl}" --output "${task_root}" \
          --protocol-dir "${source_root}/experiments/compute_matched_constraints_20pct_20261004" \
          --threads "${SLURM_CPUS_PER_TASK:-4}" --resume
      done
    done
  done
done
"${python_bin}" "${source_root}/scripts/summarize_compute_matched_constraints.py" --root "${run_root}"
