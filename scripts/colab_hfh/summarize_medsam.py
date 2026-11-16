"""Verify and tabulate the nine HFH MedSAM runs from the live status export."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(status, protocols):
    rows = []
    for seed in range(3):
        for fold in range(3):
            run = status['runs'][f'medsam3_f{fold}_s{seed}']
            done = run.get('completion', {})
            if done.get('status') != 'complete':
                continue
            if (done.get('fold'), done.get('seed'), done.get('updates'), run['saved_updates']) != (fold, seed, 200, 200):
                raise ValueError('Run identity or update count differs')
            if not run.get('evaluation_hash_verified'):
                raise ValueError('Evaluation does not match its completion hash')
            protocol = json.loads((protocols/f'f{fold}_s{seed}.json').read_text())
            scores = run['case_scores']
            row = dict(seed=seed, fold=fold, updates=200)
            for split, key, cases in [('training_support', 'training_dice', protocol['train_cases']),
                                     ('validation', 'validation_dice', protocol['evaluation_cases'])]:
                selected = [case for case in scores if case['split'] == split]
                if len(selected) != 10 or {case['case_name'] for case in selected} != set(cases):
                    raise ValueError('Evaluation cases differ from the planned split')
                values = [case['macro_hard_dice'] for case in selected]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                    raise ValueError('Invalid Dice')
                mean = statistics.mean(values)
                if not math.isclose(mean, done[key], abs_tol=1e-12, rel_tol=0):
                    raise ValueError('Reported Dice differs from the case average')
                row[key] = mean
            if len(scores) != 20:
                raise ValueError('Expected exactly 20 evaluated crops')
            row['passes_50_percent'] = row['validation_dice'] >= .5
            row['archived'] = run.get('compact_archive_receipt', {}).get('status') == 'verified'
            rows.append(row)
    result = dict(observed_at=status['observed_at'], completed=len(rows), runs=rows)
    if rows:
        result['training_mean'] = statistics.mean(row['training_dice'] for row in rows)
        result['validation_mean'] = statistics.mean(row['validation_dice'] for row in rows)
        result['validation_sd'] = statistics.stdev(row['validation_dice'] for row in rows) if len(rows) > 1 else None
        result['all_pass'] = len(rows) == 9 and all(row['passes_50_percent'] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('status', type=Path)
    parser.add_argument('--protocols', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.status.read_text()), args.protocols)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['runs']:
        with (args.output/'dice.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['runs'][0]))
            writer.writeheader(); writer.writerows(result['runs'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
