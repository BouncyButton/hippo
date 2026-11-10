"""One idempotent submission attempt plus the existing durable archive collector."""
import fcntl
import json
import subprocess
import sys
from .prepare import ROOT, NAME, REMOTE
from .campaign import read, write, sha
from scripts.cross_dataset_campaign.collect import remote, snapshot


def main():
    destination=ROOT/'reports'/NAME
    frozen=destination/'frozen'
    assert sha(frozen/'manifest.json')==read(destination/'transfer.json')['manifest_sha256']
    result=json.loads(remote(['python3',REMOTE/'campaign.py','submit','--stage',REMOTE]))
    write(destination/'submission_status.json',result)
    if result['status']=='submitted':
        with (destination/'collector.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: running=True
            else: running=False
        if not running:
            with (destination/'collector.log').open('a') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.cross_dataset_campaign.collect',
                    '--stage',str(REMOTE),'--local-stage',str(ROOT/'experiments'/NAME),
                    '--destination',str(destination)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(destination/'collector_process.json',dict(pid=proc.pid))
        state=snapshot(REMOTE);write(destination/'live.json',state)
        result['campaign_status']=state.get('status.json',{}).get('status','pending')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
