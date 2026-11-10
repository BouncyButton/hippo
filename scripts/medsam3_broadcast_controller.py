"""Deliver the staged checkpoint to this study's allocated jobs using sbcast."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    submission = json.loads((root/'SUBMISSION.json').read_text())
    pending = ({str(submission['combined_job_id']): -1} if submission.get('combined_job_id') else
               {str(job): int(fold) for fold, job in submission['jobs_by_fold'].items()})
    outcomes = {}
    while pending:
        for job, fold in list(pending.items()):
            path = root/f'job_{job}_status.json'
            if not path.exists():
                state = subprocess.check_output(['sacct', '-j', job, '-n', '-X', '-o', 'State'], text=True).strip()
                if state.startswith(('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY')):
                    outcomes[job] = dict(status='failed_before_broadcast', scheduler_state=state)
                    del pending[job]
                continue
            status = json.loads(path.read_text())
            assert status['job_id'] == job and status['fold'] == fold
            if status['status'] == 'waiting_for_base':
                destination = status['destination']
                assert destination.startswith(f'/dev/shm/medsam3-ap-{job}-') and destination.endswith('/sam3.pt')
                assert status['ready'] == str(Path(destination).parent/'ready.json')
                try:
                    subprocess.run(['sbcast', '--jobid='+job, '--force', submission['checkpoint_source'], destination], check=True, timeout=600)
                    subprocess.run(['sbcast', '--jobid='+job, '--force', str(root/'base_ready.json'), status['ready']], check=True, timeout=120)
                    outcomes[job] = dict(status='delivered', destination=destination)
                except Exception as exc:
                    outcomes[job] = dict(status='broadcast_failed', error=repr(exc))
                del pending[job]
            elif status['status'] == 'failed':
                outcomes[job] = dict(status='worker_failed', error=status.get('error'))
                del pending[job]
            elif status['status'] in ('running', 'auditing', 'complete'):
                outcomes[job] = dict(status='already_delivered')
                del pending[job]
        temporary = root/'BROADCAST_STATUS.tmp'
        temporary.write_text(json.dumps(dict(pending=pending, outcomes=outcomes), indent=2)+'\n')
        temporary.replace(root/'BROADCAST_STATUS.json')
        if pending:
            time.sleep(15)


if __name__ == '__main__':
    main()
