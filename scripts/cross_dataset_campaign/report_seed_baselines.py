"""Summarize all verified baseline seeds without discarding poor runs."""
import argparse
from pathlib import Path
from statistics import mean, stdev

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive


def report(destination):
    frozen = destination/'frozen'
    plan = read(frozen/'plan.json')
    source_sha = digest(frozen/'manifest.json')
    for name, sha in read(frozen/'manifest.json').items():
        if digest(frozen/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    rows = []
    for task in plan['tasks']:
        if task['kind'] != 'conventional' or task['arm'] != 'dice':
            raise ValueError('Only baseline campaigns are supported')
        run = destination/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        manifest = read(run/'archive_manifest.json')
        verify_archive(run, manifest)
        ack = read(run/'archive_acknowledgment.json')
        if (manifest['task'] != task or ack['manifest_sha256'] != digest(run/'archive_manifest.json')
                or ack['source_manifest_sha256'] != source_sha):
            raise ValueError('Archive/source binding changed')
        done = read(run/'completion.json')
        validation = read(run/'validation_latest.json')
        p = read(frozen/'protocols'/task['protocol']/'protocol.json')
        expected = p['models'][p['trainer_alias']]['objectives']['dice']
        if (done['status'] != 'complete' or done['updates'] != expected['max_steps']
                or done['consumed_total_flops'] > p['budget_flops']
                or (done['fold'], done['seed']) != (task['fold'], task['seed'])):
            raise ValueError('Wrong final-budget condition')
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        swaps = [r['ap_swaps'] for r in validation['cases']]
        rows.append(dict(**task, validation_dice=validation['summary']['macro_hard_dice'],
            training_dice=done['final_training_hard_dice'], best_validation_dice=done['best_validation_hard_dice'],
            foreground_fn=sum(r['foreground_fn'] for r in validation['cases']),
            foreground_fp=sum(r['foreground_fp'] for r in validation['cases']),
            ap_swaps={k: sum(r[k] for r in swaps) for k in swaps[0]} if swaps[0] is not None else None,
            updates=done['updates'], consumed_flops=done['consumed_total_flops'],
            cleanup=cleanup.get('status', 'pending'), released_file_bytes=cleanup.get('released_file_bytes', 0)))
    groups = []
    identities = sorted({(t['seed'], t['dataset'], t['model']) for t in plan['tasks']})
    for seed, dataset, model in identities:
        expected = [t for t in plan['tasks'] if (t['seed'], t['dataset'], t['model']) == (seed, dataset, model)]
        members = sorted([r for r in rows if (r['seed'], r['dataset'], r['model']) == (seed, dataset, model)], key=lambda r:r['fold'])
        values = [r['validation_dice'] for r in members]
        groups.append(dict(seed=seed, dataset=dataset, model=model, complete=len(members)==len(expected),
            completed_folds=[r['fold'] for r in members], fold_validation_dice=values,
            mean_validation_dice=mean(values) if values else None,
            sd_validation_dice=stdev(values) if len(values)>1 else None))
    seeds = []
    for seed in sorted({t['seed'] for t in plan['tasks']}):
        subset = [g for g in groups if g['seed'] == seed]
        complete = all(g['complete'] for g in subset)
        seeds.append(dict(seed=seed, complete=complete,
            mean_across_model_dataset_means=mean(g['mean_validation_dice'] for g in subset) if complete else None,
            lowest_model_dataset_mean=min(g['mean_validation_dice'] for g in subset) if complete else None,
            lowest_single_fold=min(v for g in subset for v in g['fold_validation_dice']) if complete else None))
    result = dict(status='complete' if len(rows)==len(plan['tasks']) else 'incomplete',
        completed=len(rows), total=len(plan['tasks']), groups=groups, seed_summary=seeds, runs=rows,
        released_file_bytes=sum(r['released_file_bytes'] for r in rows),
        interpretation='Final-budget hard Dice; equal-weight fold means. All requested seeds and folds retained. Seed summaries require all requested conditions. No numerical all-model success threshold or confirmatory seed selection is asserted.')
    write(destination/'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=Path)
    result = report(parser.parse_args().destination)
    print({k:result[k] for k in ('status','completed','total','seed_summary')})
