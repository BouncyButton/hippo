"""Finish this finite experiment's report once all local archives are verified."""
import json
from pathlib import Path
import subprocess
import sys
import time

from verified_archive import read, write, remote
from calibration60_pilot import NAME


def main():
    root = Path.cwd()
    report = root/'reports'/NAME
    state = report/'report_status.json'
    job = read(report/'submission.json')['job_id']
    write(state,dict(status='waiting_for_verified_archives',job_id=job))
    deadline = time.monotonic()+3*3600
    try:
        while time.monotonic()<deadline:
            progress = report/'archive_progress.json'
            if progress.exists() and read(progress).get('status')=='complete': break
            time.sleep(10)
        else: raise TimeoutError('Sweep archives did not complete; preserve recovery files')
        # Await final worker summary and local archive process bookkeeping.
        for _ in range(60):
            if (report/'summary.json').exists() and (report/'completion.json').exists(): break
            time.sleep(2)
        assert read(report/'summary.json')['status']=='complete'
        with (report/'report_generation.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/adni_constraints/report_calibration60.py')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        # BeeGFS updates personal quota counters every 30 seconds.
        time.sleep(35)
        quota = remote(['beegfs','quota','list-usage','--uids','current','--pool','pool_home']).decode()
        (report/'quota_after_cleanup.txt').write_text(quota)
        status = remote(['sacct','-j',str(job),'--format=JobID,State,ExitCode,Elapsed','--parsable2','--noheader']).decode()
        (report/'final_job_status.txt').write_text(status)
        released = sum(r['cleanup']['released_file_bytes'] for r in read(report/'archive_progress.json')['runs'])
        write(state,dict(status='complete',job_id=job,conditions=13,new_runs=13,
                         comparison='comparison.json',figure='calibration_comparison.png',
                         released_file_bytes=released,quota_after_cleanup=quota,job_status=status))
        print(json.dumps(read(state)),flush=True)
    except BaseException as error:
        write(state,dict(status='needs_attention',job_id=job,error=str(error),recovery_files_preserved=True))
        raise


if __name__=='__main__': main()
