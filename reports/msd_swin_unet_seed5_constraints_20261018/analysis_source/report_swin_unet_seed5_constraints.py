"""Paired, final-budget seed-5 constraint results against verified baselines."""
import argparse
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive


def report(destination):
    frozen = destination/'frozen'
    manifest = read(frozen/'manifest.json')
    for name, sha in manifest.items():
        if digest(frozen/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    source_sha = digest(frozen/'manifest.json')
    plan = read(frozen/'plan.json')
    baseline = {(r['model'], r['fold']): r for r in read(frozen/'baseline_results.json')}
    rows = []
    for task in plan['tasks']:
        run = destination/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        saved = read(run/'archive_manifest.json')
        verify_archive(run, saved)
        ack = read(run/'archive_acknowledgment.json')
        if (saved['task'] != task or ack['manifest_sha256'] != digest(run/'archive_manifest.json')
                or ack['source_manifest_sha256'] != source_sha):
            raise ValueError('Archive/source identity changed')
        done = read(run/'completion.json')
        p = read(frozen/'protocols'/task['protocol']/'protocol.json')
        schedule = p['models'][p['trainer_alias']]['objectives'][task['arm']]
        if (done['status'] != 'complete' or done['updates'] != schedule['max_steps']
                or done['consumed_training_flops'] > p['budget_flops']
                or (done['seed'], done['fold']) != (5, task['fold'])):
            raise ValueError('Unmatched final-budget result')
        v = read(run/'validation_latest.json')
        base = baseline[task['model'], task['fold']]
        score = v['summary']['macro_hard_dice']
        swaps = [r['ap_swaps'] for r in v['cases']]
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task, validation_dice=score, baseline_dice=base['validation_dice'],
            delta_percentage_points=100*(score-base['validation_dice']),
            best_validation_dice=done['best_validation_hard_dice'],
            foreground_fn=sum(r['foreground_fn'] for r in v['cases']),
            foreground_fp=sum(r['foreground_fp'] for r in v['cases']),
            ap_swaps={k:sum(r[k] for r in swaps) for k in swaps[0]},
            updates=done['updates'], training_flops=done['consumed_training_flops'],
            total_flops_including_calibration=done['consumed_total_flops'],
            cleanup=cleanup.get('status', 'pending'), released_file_bytes=cleanup.get('released_file_bytes', 0)))
    groups = []
    for model, arm in sorted({(t['model'], t['arm']) for t in plan['tasks']}):
        members = sorted([r for r in rows if (r['model'], r['arm']) == (model, arm)], key=lambda r:r['fold'])
        complete = len(members) == 3
        groups.append(dict(model=model, arm=arm, complete=complete,
            completed_folds=[r['fold'] for r in members],
            mean_dice=mean(r['validation_dice'] for r in members) if complete else None,
            mean_delta_percentage_points=mean(r['delta_percentage_points'] for r in members) if complete else None,
            improved_folds=[r['fold'] for r in members if r['delta_percentage_points'] > 0]))
    result = dict(status='complete' if len(rows)==len(plan['tasks']) else 'incomplete',
        completed=len(rows), total=len(plan['tasks']), baselines=list(baseline.values()),
        groups=groups, runs=rows,
        interpretation='Paired final-budget validation Dice; original optimization-only cap. Calibration separately reported. Three-fold means withheld until complete; all outcomes retained.')
    write(destination/'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    result = report(parser.parse_args().destination)
    print({k:result[k] for k in ('status', 'completed', 'total', 'groups')})
