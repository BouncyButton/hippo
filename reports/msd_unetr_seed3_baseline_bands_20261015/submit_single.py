"""One-time submission authorized by the user's 'Create a single sbatch' reply."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

stage = Path(sys.argv[1]).resolve()
expected_stage = Path('/mnt/beegfsstudents/home/3160552/msd_unetr_seed3_baseline_bands_20261015')
if stage != expected_stage:
    raise ValueError('Unexpected campaign directory')
sys.path.insert(0, str(stage/'source'))
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.unetr_seed3_job import verify

manifest_sha = '44f9dc5f509da24e0c9bcc00ea2f4b4da0ce97bbfa0800bd8279e8334d632f5a'
plan = verify(stage)
if digest(stage/'manifest.json') != manifest_sha:
    raise ValueError('Submission is bound to the reviewed single-allocation version')
preflight = read(stage/'preflight.json')
if preflight['status'] != 'passed' or preflight['manifest_sha256'] != manifest_sha:
    raise ValueError('Matching preflight required')
if not plan['single_allocation'] or 'archive_resume_module' in plan:
    raise ValueError('Only one sbatch allocation was requested')

with (stage/'submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    receipts = list((stage/'submissions').glob('*.json'))
    if receipts:
        if len(receipts) != 1:
            raise ValueError('Unexpected multiple submissions')
        print(json.dumps(dict(reused=True, **read(receipts[0]))))
        sys.exit(0)
    if read(stage/'control.json').get('pause'):
        raise ValueError('Preserve manual pause')
    current = controller.queue()
    # The exception is restricted to the exact existing chain discussed with
    # the user. It cannot bypass a different job or cancel/change a dependency.
    allowed = {'702517', '702580'}
    if (len(current) > 2 or any(row[0] not in allowed or row[2] != 'hippo-foundation-2gpu'
                               or row[1] not in ('RUNNING','COMPLETING','CONFIGURING','PENDING')
                               for row in current)):
        raise RuntimeError('Queue changed; preserve all existing jobs: '+repr(current))
    usage = controller.quota()
    if usage['free_gib'] < plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient personal quota headroom')
    dependency = ':'.join(row[0] for row in current)
    command = ['sbatch','--parsable','--chdir='+str(stage),
               '--output='+str(stage/'slurm-%j.out'),'--error='+str(stage/'slurm-%j.err')]
    if dependency:
        command += ['--dependency=afterany:'+dependency]
    command += [str(stage/'run.sbatch'),str(stage)]
    job = subprocess.check_output(command, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Ambiguous sbatch response; inspect queue before any retry: '+job)
    receipt = dict(job_id=job, dependency=dependency or None, quota=usage,
        submitted_at=time.time(), command=command, manifest_sha256=manifest_sha,
        single_allocation=True, gpu_workers=2, runs=6, prequeued_successor=False,
        queue_before_submission=current,
        authorization='Create a single sbatch — direct user reply to the request for one extra pending job behind the existing chain',
        queue_exception_scope='One additional pending allocation for these six runs only; existing jobs untouched')
    write(stage/'submissions'/f'{job}.json', receipt)
    write(stage/'submission.json', receipt)
    print(json.dumps(receipt))
