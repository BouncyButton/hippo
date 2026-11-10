"""Start this campaign's queue waiter and verified off-BeeGFS archive service."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.mni_medsam3_cluster.prepare import STAGE, REMOTE, EXPORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-stage', type=Path, default=STAGE)
    parser.add_argument('--remote-stage', type=Path, default=REMOTE)
    args = parser.parse_args()
    stage, remote, export = args.local_stage.resolve(), args.remote_stage, args.local_stage.resolve().parent
    remote_code = '''import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
stage=Path(sys.argv[1])
preflight=json.loads((stage/'preflight.json').read_text())
assert preflight['status']=='passed'
assert preflight['manifest_sha256']==sys.argv[2]==hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest()
with (stage/'admission.lock').open('a') as lock:
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print(json.dumps(dict(status='already_active',preflight=preflight)))
  sys.exit(0)
env=dict(os.environ,PYTHONPATH=str(stage/'source')+':'+str(stage/'source/scripts'),PYTHONDONTWRITEBYTECODE='1')
with (stage/'admission.log').open('a') as log:
 process=subprocess.Popen([sys.executable,'-u','-m',sys.argv[3],'admit','--stage',str(stage)],
  cwd=stage/'source',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='started',pid=process.pid,preflight=preflight,time=time.time())))
'''
    module = read(stage/'plan.json').get('controller_module', 'scripts.mni_medsam3_cluster.campaign')
    command = ['/mnt/beegfsstudents/home/3160552/.conda/envs/hippocampus/bin/python', '-c', remote_code, str(remote), digest(stage/'manifest.json'), module]
    result = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'bocconi-cluster', shlex.join(command)], text=True))
    if result['preflight']['manifest_sha256'] != digest(stage/'manifest.json'):
        raise ValueError('Remote preflight differs from local source')
    write(export/'cluster_preflight.json', result['preflight'])
    write(export/'admission_service.json', result)
    destination = export/'archive'
    destination.mkdir(exist_ok=True)
    with (destination/'collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Archive service already running')
            return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(stage/'source')+':'+str(stage/'source/scripts'))
    command = ['caffeinate', '-i', sys.executable, '-u', '-m', 'scripts.cross_dataset_campaign.collect',
        '--stage', str(remote), '--local-stage', str(stage), '--destination', str(destination)]
    with (export/'collector.log').open('a') as log:
        process = subprocess.Popen(command, cwd=stage/'source', env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(export/'collector_service.json', dict(pid=process.pid, command=command, time=time.time()))
    print(json.dumps(dict(admission=result['status'], collector_pid=process.pid, archive=str(destination))))


if __name__ == '__main__':
    main()
