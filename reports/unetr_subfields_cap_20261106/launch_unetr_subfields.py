"""Check access/queue/quota, deploy the frozen UNETR campaign, and collect it."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from scripts.unetr_subfields.contracts import ROOT, NAME
from scripts.unetr_subfields.job import verify
from scripts.cross_dataset_campaign.common import read, write, digest, REMOTE

HOST = 'bocconi-cluster'


def remote(arguments, timeout=180):
    return subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
         shlex.join([str(value) for value in arguments])], stderr=subprocess.PIPE,
        text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    stage = ROOT / 'experiments' / NAME
    report = ROOT / 'reports' / NAME
    collector_python = ROOT / '.venv' / 'bin' / 'python'
    if not collector_python.exists():
        collector_python = Path(sys.executable)
    subprocess.run([str(collector_python), '-c',
        'import inspect, tarfile; assert "filter" in inspect.signature(tarfile.TarFile.extractall).parameters, '
        '"The archive collector requires Python with tarfile extraction filters"'], check=True)
    plan = verify(stage)
    destination = REMOTE / NAME
    report.mkdir(parents=True, exist_ok=True)
    try:
        queue_text = remote(['squeue', '-h', '-u', '3160552', '-o', '%i|%T|%j'])
        quota_text = remote(['beegfs', 'quota', 'list-usage', '--uids', 'current', '--pool', 'pool_home'])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        message = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        write(report / 'access_status.json', dict(status='blocked_cluster_access', host=HOST,
            error=message, checked_at=time.time(), submitted=False,
            manifest_sha256=digest(stage / 'manifest.json')))
        print(message, file=sys.stderr)
        return 1
    queue = [row.split('|') for row in queue_text.splitlines() if row.strip()]
    match = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', quota_text)
    if not match:
        raise RuntimeError('Cannot verify personal quota')
    used, limit = map(float, match.groups())
    write(report / 'access_status.json', dict(status='connected', host=HOST, queue=queue,
        quota=dict(used_gib=used, limit_gib=limit, free_gib=limit-used, raw=quota_text),
        checked_at=time.time(), submitted=False))
    if len(queue) > 1 or any(row[1] not in ('RUNNING', 'COMPLETING', 'CONFIGURING') for row in queue):
        raise RuntimeError('No slot under the one-running/one-queued policy')
    if limit - used < plan['storage']['minimum_free_gib']:
        raise RuntimeError('Insufficient verified storage headroom; no job submitted')
    if args.check_only:
        print('Cluster reachable; queue and quota checks passed')
        return 0
    # Refuse to overwrite a different frozen campaign, including active work.
    code = ('import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); '
            'm=p/"manifest.json"; '
            'assert not p.exists() or (m.exists() and hashlib.sha256(m.read_bytes()).hexdigest()==sys.argv[2]), "Remote stage differs"; '
            'p.mkdir(parents=True,exist_ok=True)')
    remote(['python3', '-c', code, destination, digest(stage / 'manifest.json')])
    subprocess.run(['rsync', '-az', '--exclude=local_preflight.json', '--exclude=__pycache__',
                    str(stage) + '/', f'{HOST}:{destination}/'], check=True)
    python = plan['conventional_python']
    environment = ':'.join(str(destination / name) for name in ('source', 'source/scripts', 'source/baselines/nnUNet'))
    code = ('import os,sys; from pathlib import Path; os.chdir(sys.argv[1]); '
            'from scripts.unetr_subfields.job import preflight; preflight(Path(sys.argv[2]))')
    remote(['env', 'PYTHONPATH=' + environment, python, '-c', code,
            destination / 'source', destination], timeout=600)
    # The GPU worker will wait for this collector's verified archive receipts.
    log = (report / 'collector.log').open('a')
    collector = subprocess.Popen(['caffeinate', '-i', str(collector_python), '-u', '-m',
        'scripts.cross_dataset_campaign.collect', '--stage', str(destination),
        '--local-stage', str(stage), '--destination', str(report)], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    write(report / 'collector_process.json', dict(pid=collector.pid, python=str(collector_python), started_at=time.time()))
    code = ('import json,os,sys; from pathlib import Path; os.chdir(sys.argv[1]); '
            'from scripts.unetr_subfields import job; '
            'job.controller.verify=job.verify; '
            'print(json.dumps(job.controller.submit(Path(sys.argv[2]))))')
    output = remote(['env', 'PYTHONPATH=' + environment, python, '-c', code,
                     destination / 'source', destination])
    import json
    receipt = json.loads(output)
    write(report / 'submission.json', receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
