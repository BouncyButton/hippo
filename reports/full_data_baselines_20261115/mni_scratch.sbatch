#!/usr/bin/env bash
#SBATCH --job-name=full-MNI-scratch
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode04
set -Eeuo pipefail
root=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
stage="$root/full_data_baselines_20261115"
bundle="$stage/mni_scratch"
py="$root/.conda/envs/hippocampus/bin/python"
export PYTHONPATH="$bundle/compatibility:$bundle/source:$bundle/source/scripts:$bundle/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-full-mpl-$SLURM_JOB_ID" WANDB_MODE=disabled
cd "$bundle"
mapfile -t ordered < <(python3 -c 'import json; print("\n".join(json.load(open("cells.json"))))')
test "${#ordered[@]}" -eq 36
for key in "${ordered[@]}"; do
    output="$stage/runs/$key"
    if test -f "$output/cleanup.json" && "$py" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="complete"' "$output/cleanup.json"; then
        continue
    fi
    mkdir -p "$output"
    if ! test -e "$output/completion.json"; then
        "$py" -c 'from scripts.cross_dataset_campaign.storage import quota; assert quota()["free_gib"]>=12'
        "$py" -c 'import json,sys,os; print(json.dumps(dict(key=sys.argv[1],job=os.environ["SLURM_JOB_ID"])))' "$key" > "$output/allocation.json"
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 15 > "$output/gpu.csv" &
        monitor=$!
        trap 'kill "$monitor" 2>/dev/null || true' EXIT
        srun --exclusive --exact -n1 --cpus-per-task=4 --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$py" -u "$bundle/worker.py" "$key" --bundle "$bundle" --work "$stage/runs" >> "$output/training.log" 2>&1
        kill "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
        trap - EXIT
    fi
    "$py" "$stage/mni_scratch_archive.py" "$bundle" "$key"
    # Retire each completed cell as soon as the local collector verifies it.
    for ((n=0; n<120; n++)); do
        test ! -f "$output/cleanup.json" || break
        sleep 5
    done
    test -f "$output/cleanup.json" || exit 75
done
