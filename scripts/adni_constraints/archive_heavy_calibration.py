"""Local archive/acknowledgment half of the finite 12-run experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from verified_archive import archive_one, remote, read, write
from heavy_calibration_pilot import NAME, REMOTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    report = stage.parents[1]/'reports'/NAME
    remote_stage = REMOTE/NAME
    remote_results = Path(str(remote_stage)+'_results')
    plan = read(stage/'plan.json')
    completed = []
    for spec in plan['runs']:
        arm = spec['objective']
        output = remote_results/arm
        destination = report/'results'/arm
        while True:
            # Each SSH read is bounded, no job is resubmitted by this process.
            query = ['python3', '-c',
                'import json,pathlib; p=pathlib.Path('+repr(str(output))+'); '
                'print(json.dumps({n:(p/n).exists() for n in ["archive_manifest.json","cleanup.json"]}))']
            state = json.loads(remote(query))
            if state['archive_manifest.json']: break
            time.sleep(10)
        archive_one(output, destination, report/'source_archive')
        while True:
            query = ['python3', '-c',
                'import json,pathlib; p=pathlib.Path('+repr(str(output/'cleanup.json'))+'); '
                'print(p.read_text() if p.exists() else "{}")']
            cleanup = json.loads(remote(query))
            if cleanup.get('status') == 'complete': break
            time.sleep(5)
        write(destination/'cleanup.json', cleanup)
        completed.append(dict(objective=arm, final_dice=read(destination/'verified_result.json')['final_dice'],
                              cleanup=cleanup))
        write(report/'archive_progress.json', dict(status='complete' if len(completed)==12 else 'in_progress',
              completed=len(completed), runs=completed))
        print(json.dumps(dict(archived=arm, final_dice=completed[-1]['final_dice'],
                              deleted_file_bytes=cleanup['released_file_bytes'])), flush=True)
    # The worker writes its summary just after each cleanup receipt.
    for _ in range(30):
        try:
            summary = json.loads(remote(['cat',str(remote_results/'summary.json')]))
            if summary['status']=='complete': break
        except subprocess.CalledProcessError: pass
        time.sleep(2)
    assert summary['status']=='complete'
    write(report/'summary.json', summary)
    for filename in ('launch.json',):
        write(report/filename,json.loads(remote(['cat',str(remote_results/filename)])))
    for filename in ('submission.json','completion.json'):
        write(report/filename,json.loads(remote(['cat',str(remote_stage/filename)])))
    job = read(report/'submission.json')['job_id']
    for extension in ('out','err'):
        (report/f'slurm-{job}.{extension}').write_bytes(remote([
            'cat',str(remote_stage/f'slurm-hippo-adni-nnunet-target40-{job}.{extension}')]))
    print(json.dumps(dict(status='complete',completed=12)),flush=True)


if __name__=='__main__': main()
