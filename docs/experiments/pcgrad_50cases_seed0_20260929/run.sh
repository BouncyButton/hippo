#!/usr/bin/env bash
#SBATCH --job-name=pcgrad-50cases-s0
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
ROOT="${1:?Pass experiment root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/source"
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
record_exit() {
 local code=$?
 trap - EXIT
 "$PYTHON" - "$ROOT" "$code" <<'EXITPY'
import json,os,sys,datetime
from pathlib import Path
(Path(sys.argv[1])/'LAUNCHER_EXIT.json').write_text(json.dumps(dict(job_id=os.getenv('SLURM_JOB_ID'),exit_code=int(sys.argv[2]),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
EXITPY
 exit "$code"
}
trap record_exit EXIT
cd "$ROOT/source"
TEST_RUNTIME=/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01/test_runtime
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TEST_RUNTIME:$ROOT/source" "$PYTHON" -m pytest -q thesis/new_constraints/test_pcgrad.py > "$ROOT/GPU_TESTS.txt" 2>&1
"$PYTHON" scripts/train_pcgrad_50cases.py --root "$ROOT" --mode train
