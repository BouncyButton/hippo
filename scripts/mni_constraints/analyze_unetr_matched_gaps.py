"""Describe training/validation gap changes in verified matched results."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from scripts.cross_dataset_campaign.common import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--result-file', required=True)
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[2] / 'reports' / args.name
    raw = (report / args.result_file).read_bytes()
    result = json.loads(raw)
    rows = []
    for item in result['rows']:
        baseline, current = item['baseline'], item['current']
        baseline_gap = baseline['training_percent'] - baseline['validation_percent']
        current_gap = current['training_percent'] - current['validation_percent']
        change = current_gap - baseline_gap
        assert abs(change - item['training_gain_pp'] + item['gain_pp']) < 1e-9
        rows.append(dict(
            candidate=item['candidate'], seed=item['seed'], fold=item['fold'],
            baseline_gap_pp=baseline_gap, constraint_gap_pp=current_gap,
            gap_change_pp=change, training_gain_pp=item['training_gain_pp'],
            validation_gain_pp=item['gain_pp'],
            baseline_archive_sha256=baseline['archive_manifest_sha256'],
            constraint_archive_sha256=current['archive_manifest_sha256'],
        ))
    summary = dict(
        pairs=len(rows),
        unique_seed_fold_pairs=len({(row['seed'], row['fold']) for row in rows}),
        candidate_pairs={candidate: sum(row['candidate'] == candidate for row in rows)
                         for candidate in sorted({row['candidate'] for row in rows})},
        wider_gap_pairs=sum(row['gap_change_pp'] > 0 for row in rows),
        narrower_gap_pairs=sum(row['gap_change_pp'] < 0 for row in rows),
        unchanged_gap_pairs=sum(row['gap_change_pp'] == 0 for row in rows),
    )
    output = dict(
        checked_at=time.time(),
        input_result_sha256=hashlib.sha256(raw).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows, **summary,
        scope='Final-budget matched training and validation macro hard Dice, in percentage points.',
        caveat='Counts pool candidates and are not a nine-pair confirmation. Descriptive development comparisons with overlapping folds; gap changes alone do not establish a causal mechanism or independent statistical evidence.',
    )
    write(report / 'matched_gap_diagnostics.json', output)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
