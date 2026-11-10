"""Optional local archiving for a bounded study; never submits GPU jobs."""
import argparse
import json
from pathlib import Path
import time
import traceback

from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--local-stage', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    deadline = read(args.local_stage/'plan.json')['deadline']
    while True:
        try:
            collect.serve(args.stage, args.local_stage, args.destination, once=True)
            state = read(args.destination/'live.json').get('status.json', {})
            terminal = state.get('status') in (
                'complete', 'complete_pending_archive', 'failed', 'paused',
                'deadline_admission_cutoff', 'deadline_reached', 'storage_headroom_limit')
            if terminal:
                # The worker may have exited before the final archive receipt.
                # Retire only acknowledged completed runs after it releases its lock.
                code = '''import fcntl,json,sys
from pathlib import Path
from scripts.cross_dataset_campaign.common import read
from scripts.cross_dataset_campaign.storage import retire,quota
p=Path(sys.argv[1]);cleaned=[]
with (p/'worker.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for d in (p/'runs').iterdir():
  if (d/'completion.json').exists() and (d/'archive_acknowledgment.json').exists() and read(d/'completion.json')['status']=='complete':
   if not (d/'cleanup.json').exists() or read(d/'cleanup.json')['status']!='complete':retire(d)
   cleaned.append(d.name)
 result=dict(cleaned=cleaned,quota=quota())
 for name in ('refinement_results.json','bounded_completion.json'):
  if (p/name).exists():result[name]=read(p/name)
 print(json.dumps(result))'''
                env = ':'.join(str(args.stage/p) for p in ('source', 'source/scripts', 'source/baselines/nnUNet'))
                result = json.loads(collect.remote(['env', 'PYTHONPATH='+env, 'python3', '-c', code, args.stage]))
                write(args.destination/'final_archive_cleanup.json', result)
                collect.serve(args.stage, args.local_stage, args.destination, once=True)
                return
        except Exception:
            write(args.destination/'bounded_collector_error.json', dict(time=time.time(), traceback=traceback.format_exc()))
        # Laptop sleep/network loss never extends the experiment or this service.
        if time.time() > deadline+900:
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
