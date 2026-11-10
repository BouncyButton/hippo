"""Submit an isolated, bounded runtime benchmark while the scientific sweep is held."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from scripts.cross_dataset_campaign.common import digest,write,REMOTE
from scripts.cross_dataset_campaign.msd_mixed_losses_job import NAME,ROOT


def submit(name,startup=False):
    stage=REMOTE/NAME;bench=REMOTE/name
    files=('preload_msd_runtime','runtime_worker','benchmark_msd_startup') if startup else ('msd_runtime_fast','benchmark_msd_runtime')
    sources={f'scripts/cross_dataset_campaign/{n}.py':digest(ROOT/'scripts/cross_dataset_campaign'/(n+'.py')) for n in files}
    module='scripts.cross_dataset_campaign.benchmark_msd_startup' if startup else 'scripts.cross_dataset_campaign.benchmark_msd_runtime'
    manifest=dict(parent_manifest_sha256=digest(ROOT/'experiments'/NAME/'manifest.json'),
        additional_sources=sources,authorization='User requested optimizing whatever is possible before full sweep; bounded equivalence and runtime benchmark only',
        time_limit_minutes=30,minimum_free_gib=24)
    batch=f'''#!/usr/bin/env bash
#SBATCH --job-name=hippo-msd-runtime-bench
#SBATCH --partition=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
set -Eeuo pipefail
export PYTHONPATH="{stage}/source:{stage}/source/scripts:{stage}/source/baselines/nnUNet"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 nnUNet_compile=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="/tmp/hippo-runtime-bench-${{SLURM_JOB_ID}}"
cd "{stage}/source"
/home/3160552/.conda/envs/hippocampus/bin/python -m {module} --stage "{stage}" --output "{bench}/runs"
'''
    code='''import json,sys,subprocess,time
from pathlib import Path
request=json.load(sys.stdin);stage=Path(request['stage']);bench=Path(request['bench']);manifest=request['manifest']
sys.path[:0]=[str(stage/'source'),str(stage/'source/scripts'),str(stage/'source/baselines/nnUNet')]
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.msd_mixed_losses_job import verify
from scripts.cross_dataset_campaign.storage import quota
from scripts.cross_dataset_campaign.common import write,digest,read
verify(stage)
if not read(stage/'control.json')['pause']:raise ValueError('Full sweep must remain held')
for name,sha in manifest['additional_sources'].items():
 if digest(stage/'source'/name)!=sha:raise ValueError('Benchmark source hash differs')
usage=quota()
if usage['free_gib']<manifest['minimum_free_gib']:raise ValueError('Insufficient benchmark headroom')
bench.mkdir(exist_ok=True)
if (bench/'submission.json').exists():
 print(json.dumps(read(bench/'submission.json')));sys.exit(0)
active=controller.queue()
if active:raise RuntimeError('Benchmark requires idle account: '+repr(active))
write(bench/'manifest.json',manifest);(bench/'run.sbatch').write_text(request['batch'])
subprocess.run(['bash','-n',str(bench/'run.sbatch')],check=True)
job=subprocess.check_output(['sbatch','--parsable','--chdir='+str(bench),'--output='+str(bench/'slurm-%j.out'),'--error='+str(bench/'slurm-%j.err'),str(bench/'run.sbatch')],text=True).strip().split(';')[0]
if not job.isdigit():raise ValueError('Unexpected sbatch output')
receipt=dict(job_id=job,submitted_at=time.time(),quota=usage,benchmark_only=True,full_sweep_held=True,manifest_sha256=digest(bench/'manifest.json'))
write(bench/'submission.json',receipt);print(json.dumps(receipt))
'''
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',
        shlex.join(['python3','-c',code])],input=json.dumps(dict(stage=str(stage),bench=str(bench),manifest=manifest,batch=batch)).encode())
    receipt=json.loads(raw);report=ROOT/'reports'/name
    write(report/'submission.json',receipt);write(report/'manifest.json',manifest)
    (report/'run.sbatch').write_text(batch)
    for relative in sources:
        dest=report/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--startup',action='store_true')
    args=parser.parse_args()
    if not args.name.startswith('msd_runtime_benchmark_') or '/' in args.name:raise ValueError('Invalid benchmark name')
    submit(args.name,args.startup)
