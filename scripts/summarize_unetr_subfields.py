"""Summarize all predeclared runs, without replacing poor seeds or folds."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from scripts.unetr_subfields.contracts import ROOT, NAME, DATASETS, tasks
from scripts.cross_dataset_campaign.common import read, write, digest


def summarize(folder):
    rows, missing = [], []
    for task in tasks():
        run = folder / 'runs' / task['id']
        if not (run / 'validation_latest.json').exists() or not (run / 'cleanup.json').exists():
            missing.append(task['id'])
            continue
        if read(run / 'cleanup.json')['status'] != 'complete':
            missing.append(task['id'])
            continue
        done = read(run / 'completion.json')
        result = read(run / 'validation_latest.json')
        rows.append(dict(dataset=task['dataset'], fold=task['fold'], seed=task['seed'],
            foreground_subfield_dice=result['summary']['macro_hard_dice'],
            whole_hippocampus_dice=result['summary']['whole_hippocampus_hard_dice'],
            best_validation_dice=done['best_validation_hard_dice'], updates=done['updates'],
            consumed_training_flops=done['consumed_training_flops'],
            foreground_fn=sum(case['foreground_fn'] for case in result['cases']),
            foreground_fp=sum(case['foreground_fp'] for case in result['cases'])))
    groups = {}
    for dataset in DATASETS:
        selected = [row for row in rows if row['dataset'] == dataset]
        scores = [row['foreground_subfield_dice'] for row in selected]
        groups[dataset] = dict(completed=len(scores), expected=9, complete=len(scores) == 9)
        if scores:
            groups[dataset].update(mean=statistics.mean(scores),
                sd=statistics.stdev(scores) if len(scores) > 1 else None,
                minimum=min(scores), maximum=max(scores),
                runs_at_least_050=sum(score >= .5 for score in scores),
                mean_in_descriptive_045_055_band=.45 <= statistics.mean(scores) <= .55,
                fold_means={str(fold): statistics.mean([row['foreground_subfield_dice'] for row in selected if row['fold'] == fold])
                            for fold in range(3) if any(row['fold'] == fold for row in selected)})
    report = dict(status='complete' if not missing else 'incomplete', groups=groups, runs=rows, missing_runs=missing,
                  primary='Final-budget foreground-subfield macro hard Dice. Best validation is secondary.',
                  inference='Exploratory first-three-of-five folds; report all predeclared seeds. No independent test set.',
                  source_sha256=digest(Path(__file__)))
    write(folder / 'summary.json', report)
    if rows:
        with (folder / 'runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=ROOT / 'reports' / NAME)
    args = parser.parse_args()
    print(json.dumps(summarize(args.report), indent=2))
