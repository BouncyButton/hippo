#!/usr/bin/env bash
#SBATCH --job-name=pcgrad-edge-f0-3seeds
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
(Path(sys.argv[1])/'LAUNCHER_EXIT.json').write_text(json.dumps(dict(job_id=os.getenv('SLURM_JOB_ID'),exit_code=int(sys.argv[2]),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
EXITPY
 exit "$code"
}
trap record_exit EXIT
cd "${ROOT}/source"
"$PYTHON" - "$ROOT" <<'QUOTAPY'
from pathlib import Path
import sys,torch
sys.path.insert(0,'scripts')
from run_degree_bands_ab import check_quota
assert torch.cuda.is_available()
check_quota(Path(sys.argv[1]),'allocation_start',1.6)
QUOTAPY
TEST_RUNTIME=/mnt/beegfsstudents/home/3160552/separated_edge_20260929_01/test_runtime
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$TEST_RUNTIME:$ROOT/source" "$PYTHON" -m pytest -q thesis/new_constraints/test_pcgrad.py > "$ROOT/GPU_TESTS.txt" 2>&1
for SEED in 0 1 2; do
 "$PYTHON" scripts/train_pcgrad_edge.py --root "$ROOT/fold0/seed$SEED" --seed "$SEED" --mode audit
done
"$PYTHON" - "$ROOT" <<'GATEPY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
reports=[json.loads((root/f'fold0/seed{s}/all_case_conflicts.json').read_text()) for s in range(3)]
assert all(r['status']=='complete' and len(r['cases'])==10 for r in reports)
gate=dict(seeds=[0,1,2],passed=all(r['confirmed'] for r in reports),counts=[r['inner_cross_conflicting_cases'] for r in reports],scope='Majority inner-cross conflict in each seed, using all ten training cases at the old selected separated-control checkpoints.')
(root/'CONFLICT_GATE.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate),flush=True)
if not gate['passed']:raise SystemExit('Conflict confirmation failed; no new models will be fitted.')
GATEPY
for SEED in 0 1 2; do
 "$PYTHON" scripts/train_pcgrad_edge.py --root "$ROOT/fold0/seed$SEED" --seed "$SEED" --mode train
done
"$PYTHON" scripts/summarize_pcgrad_edge.py "$ROOT"
