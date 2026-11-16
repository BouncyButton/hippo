"""Audit every downloaded strict-search archive against its frozen control."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from scripts.colab_hfh.bands_strict_search import qualifies


def verify(path: Path, references: dict) -> dict:
    with zipfile.ZipFile(path) as archive:
        def read(name):
            return json.loads(archive.read(name))

        manifest = read('results/archive_manifest.json')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read('results/' + name)).hexdigest() != digest:
                raise ValueError(f'Archive integrity failure: {path.name}/{name}')
        row = read('results/paired_comparison.json')
        key = f"{row['model']}_f{row['fold']}_s{row['seed']}"
        reference = references[key]
        config = read('results/config.json')
        protocol = read('protocol/protocol.json')
        for field in ('model_identity', 'seed', 'fold', 'dataset_sha256',
                      'train_cases', 'validation_cases', 'lr_horizon_updates',
                      'initial_lr', 'augmentation', 'precision', 'max_steps'):
            if config[field] != reference['configuration'][field]:
                raise ValueError(f'Unmatched {field}: {key}')
        if read('results/initialization.json') != reference['initialization']:
            raise ValueError(f'Unmatched initialization: {key}')
        if row['baseline_archive_sha256'] != reference['archive_sha256']:
            raise ValueError(f'Unmatched control archive: {key}')
        calibration = read('results/calibration.json')
        if [case['case_name'] for case in calibration['cases']] != config['train_cases']:
            raise ValueError(f'Calibration cases differ: {key}')
        if set(config['train_cases']) & set(config['validation_cases']):
            raise ValueError(f'Train/validation overlap: {key}')
        if protocol.get('band_reset_momentum'):
            transition = read('results/band_optimizer_reset.json')
            start = protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']['prefix_updates']
            if (transition['attempt'] != start + 1 or transition['cleared_state_entries'] <= 0
                    or row.get('training_transition') != transition['policy']):
                raise ValueError(f'Optimizer transition mismatch: {key}')
        score = read('results/validation_latest.json')['summary']['macro_hard_dice']
        baseline = reference['result']['validation_dice']
        if row['band_dice'] != score or row['baseline_dice'] != baseline:
            raise ValueError(f'Reported Dice differs from source: {key}')
        if abs(row['delta_dice_pp'] - 100 * (score - baseline)) > 1e-9:
            raise ValueError(f'Incorrect paired difference: {key}')
        ledger = read('results/budget_ledger.json')
        if ledger['training_flops'] != row['optimization_flops'] or ledger['training_flops'] > protocol['budget_flops']:
            raise ValueError(f'Optimization budget violation: {key}')
        row.update(recipe=path.parent.name, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--payload', type=Path, default=Path('reports/hfh_colab_exploration_20261111/bands_v1_payload.json'))
    args = parser.parse_args()
    references = json.loads(args.payload.read_text())['references']
    rows = [verify(path, references) for path in sorted(args.directory.glob('*/*.zip'))]
    summaries = []
    for recipe in sorted({row['recipe'] for row in rows}):
        group = [row for row in rows if row['recipe'] == recipe]
        summaries.append(dict(recipe=recipe, model=group[0]['model'], pairs=len(group),
                             kind='optimizer_control' if any(row.get('control_kind') for row in group) else 'BAND',
                             qualified=qualifies(group),
                             minimum_delta_pp=min(row['delta_dice_pp'] for row in group),
                             mean_delta_pp=statistics.mean(row['delta_dice_pp'] for row in group)))
    result = dict(criterion='Every one of nine fixed pairs >= +1.0 Dice pp; all Dice >=50%',
                  interpretation='Validation-selected exploration; incomplete candidates are not qualified',
                  verified_pairs=len(rows),
                  search_optimization_flops=sum(row['optimization_flops'] for row in rows),
                  search_calibration_flops=sum(row['calibration_flops'] for row in rows),
                  candidates=summaries)
    (args.directory / 'verified_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if rows:
        with (args.directory / 'verified_pairs.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
