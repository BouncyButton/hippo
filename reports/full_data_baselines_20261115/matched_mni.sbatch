#!/usr/bin/env bash
#SBATCH --job-name=full-MNI-matched
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4g.40gb:2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode04
set -Eeuo pipefail
base=${HIPPO_CLUSTER_ROOT:?Set HIPPO_CLUSTER_ROOT}
stage="$base/full_data_baselines_20261115"
cd "$stage/matched_MNI"
py="$base/.conda/envs/hippocampus/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mapfile -t keys < <("$py" -c 'import json;print("\n".join(k for k,r in json.load(open("jobs.json")).items() if r["feasible"]))')
test "${#keys[@]}" -eq 36
run_lane() (
  local lane="$1"
  for ((i=lane; i<${#keys[@]}; i+=2)); do
    key="${keys[$i]}"
    if [[ -f "work/$key/durable_ack.json" ]]; then continue; fi
    "$py" -c "import sys;sys.path.insert(0,'$base/mni_medsam3_n5_f0_s0_400_20261110/source');from scripts.cross_dataset_campaign.storage import quota;assert quota()['free_gib']>=12"
    srun --exact --exclusive -n1 --cpus-per-task=4 --mem=40G --gres=gpu:4g.40gb:1 --gpus-per-task=4g.40gb:1 --gpu-bind=single:1 "$py" -u run_matched_controls.py --original "$stage/mni_scratch" "$key"
    for ((n=0; n<120; n++)); do
        [[ -f "work/$key/durable_ack.json" ]] && break
        sleep 5
    done
    [[ -f "work/$key/durable_ack.json" ]] || exit 75
  done
)
"$py" -c "import sys;sys.path.insert(0,'$base/mni_medsam3_n5_f0_s0_400_20261110/source');from scripts.cross_dataset_campaign.storage import quota;assert quota()['free_gib']>=24"
run_lane 0 &
first=$!
run_lane 1 &
second=$!
status=0
wait "$first" || status=$?
wait "$second" || status=$?
exit "$status"
