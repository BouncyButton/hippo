#!/usr/bin/env bash
#SBATCH --job-name=edge-rules-f0-3seeds
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail
ROOT="${1:?Pass experiment root}"
export PYTHONPATH="${ROOT}/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PYTHON=/home/3160552/.conda/envs/hippocampus/bin/python
record_exit() {
  local code=$?
  trap - EXIT
  "$PYTHON" - "$ROOT" "$code" <<'EXITPY'
import datetime,json,os,sys
from pathlib import Path
(Path(sys.argv[1])/'LAUNCHER_EXIT.json').write_text(json.dumps(dict(
    job_id=os.getenv('SLURM_JOB_ID'),exit_code=int(sys.argv[2]),
    finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
EXITPY
  exit "$code"
}
trap record_exit EXIT
cd "${ROOT}/source"
"$PYTHON" - "$ROOT" <<'QUOTAPY'
from pathlib import Path
import sys
sys.path.insert(0,'scripts')
from run_degree_bands_ab import check_quota
check_quota(Path(sys.argv[1]),'allocation_start',1.6)
QUOTAPY
# Intentionally restricted to ONE fold. Expansion requires reviewing this pilot.
for SEED in 0 1 2; do
  "$PYTHON" scripts/train_separated_edge.py --root "${ROOT}/fold0/seed${SEED}" --fold 0 --seed "$SEED"
done
"$PYTHON" scripts/summarize_separated_edge.py "$ROOT"
