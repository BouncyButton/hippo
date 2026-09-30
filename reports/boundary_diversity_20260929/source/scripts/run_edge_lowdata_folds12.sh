#!/usr/bin/env bash
#SBATCH --job-name=edge-lowdata-folds12
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
ROOT="${1:?Pass experiment root}"
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
record_exit() {
  local code=$?
  trap - EXIT
  "$PYTHON" - "$ROOT" "$code" <<'PY'
import datetime, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
(root/'LAUNCHER_EXIT.json').write_text(json.dumps({
    'job_id': os.getenv('SLURM_JOB_ID'), 'exit_code': int(sys.argv[2]),
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
PY
  exit "$code"
}
trap record_exit EXIT
cd "${ROOT}/source"
for FOLD in 1 2; do
  for SEED in 0 1 2; do
    "$PYTHON" scripts/train_edge_lowdata.py --root "${ROOT}/fold${FOLD}/seed${SEED}" \
      --fold "$FOLD" --seed "$SEED" --compact-checkpoints --minimum-free-gib 0.8
  done
  "$PYTHON" scripts/summarize_edge_lowdata.py --root "${ROOT}/fold${FOLD}" --fold "$FOLD"
done
"$PYTHON" - "$ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
summaries = {str(fold): json.loads((root/f'fold{fold}/SUMMARY.json').read_text()) for fold in (1, 2)}
assert all(value['status'] == 'complete' for value in summaries.values())
(root/'SUMMARY.json').write_text(json.dumps({'status': 'complete', 'folds': summaries}, indent=2))
PY
