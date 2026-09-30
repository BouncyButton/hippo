#!/usr/bin/env bash
#SBATCH --job-name=degree-edge-A-retry
#SBATCH --partition=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
ROOT="${1:?Pass retry root}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
record_exit() {
  local code="$?"
  trap - EXIT
  python3 - "$ROOT" "$code" <<'PY'
import datetime, json, os, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
record = {'job_id': os.environ.get('SLURM_JOB_ID'), 'exit_code': int(sys.argv[2]),
          'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(json.dumps({'launcher_exit': record}), flush=True)
(root / 'LAUNCHER_EXIT.json').write_text(json.dumps(record, indent=2) + '\n')
quota = subprocess.run(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--gids', 'current'],
                       capture_output=True, text=True, timeout=30)
(root / 'quota_exit.txt').write_text(quota.stdout + quota.stderr)
PY
  exit "$code"
}
trap record_exit EXIT
bash "$ROOT/source/scripts/run_degree_edge_lowdata.sh" "$ROOT"
