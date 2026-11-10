"""Maintain a local record of every archived seed attempt and selection decision."""
import argparse
import csv
from pathlib import Path
import time

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.unetr_subfields.seed_search import ROOT, NAME, selection, THRESHOLD, TARGET


def summarize(folder):
    live = read(folder / 'live.json') if (folder / 'live.json').exists() else {}
    state = live.get('status.json', {})
    rows = []
    for run in sorted((folder / 'runs').glob('*')):
        if run.name.startswith('abandoned_'):
            continue
        if not (run / 'cleanup.json').exists() or read(run / 'cleanup.json')['status'] != 'complete':
            continue
        manifest = read(run / 'archive_manifest.json')
        verify_archive(run, manifest)
        task = manifest['task']
        done, evaluation = read(run / 'completion.json'), read(run / 'validation_latest.json')
        if done['status'] != 'complete' or done['consumed_training_flops'] > done['budget_flops']:
            raise ValueError('Only completed in-budget runs can enter selection')
        rows.append(dict(dataset=task['dataset'], objective=task['arm'], seed=task['seed'], fold=task['fold'],
            final_dice=evaluation['summary']['macro_hard_dice'],
            whole_hippocampus_dice=evaluation['summary']['whole_hippocampus_hard_dice'],
            best_dice=done['best_validation_hard_dice'], updates=done['updates'],
            consumed_training_flops=done['consumed_training_flops'],
            foreground_fn=sum(c['foreground_fn'] for c in evaluation['cases']),
            foreground_fp=sum(c['foreground_fp'] for c in evaluation['cases']),
            released_file_bytes=read(run / 'cleanup.json')['released_file_bytes']))
    result = selection(rows, state.get('seed_search', {}).get('active_datasets'))
    summary = dict(status=state.get('status', 'not_started'),
        outcome=state.get('search_outcome', 'searching'), active=state.get('active'),
        completed=len(rows), runs=rows, **result,
        released_file_bytes=sum(r['released_file_bytes'] for r in rows),
        last_observed_quota=state.get('quota'), source_sha256=digest(Path(__file__)),
        updated_at=time.time())
    write(folder / 'summary.json', summary)
    if rows:
        with (folder / 'runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports' / NAME)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    previous = None
    while True:
        result = summarize(args.report)
        marker = (result['status'], result['completed'], result['active'])
        if marker != previous:
            print(dict(status=result['status'], completed=result['completed'], active=result['active'],
                qualifying_seeds={d:g['qualifying_seeds'] for d,g in result['groups'].items()}), flush=True)
            previous = marker
        if not args.watch or result['status'] in ('complete', 'failed', 'paused'):
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
