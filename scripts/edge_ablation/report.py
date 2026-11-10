"""Paired validation comparisons against Dice+bands and between matched factors."""
import csv
import json
import math
from scripts.depth_supervision.utils import atomic_json, write_history
from .train import completed_record, identity


def report(p, root):
    records, cases, rows, configs, histories = {}, {}, [], {}, {}
    for entry in p['runs']:
        folder = root/'runs'/entry['id']
        record = completed_record(folder, identity(p, entry))
        if record is None:
            continue
        name = entry['id']
        records[name] = record
        configs[name] = json.loads((folder/'config.json').read_text())
        with (folder/'metrics.csv').open() as stream:
            histories[name] = list(csv.DictReader(stream))
        cases[name] = {r['case_name']: r for r in json.loads((folder/'validation_best.json').read_text())['cases']}
        rows.append(dict(objective=name, best_epoch=record['best_epoch'],
                         completed_epochs=record['completed_epochs'],
                         training_seconds=record['training_seconds'], **record['validation']))
    comparisons, factors = {}, []
    def compare(name, base):
        assert set(cases[name]) == set(cases[base])
        paired = []
        for case, row in cases[name].items():
            other = cases[base][case]
            delta = {key: row[key]-other[key] for key in records[name]['validation']
                     if key in row and key in other and isinstance(row[key], (int, float)) and isinstance(other[key], (int, float))}
            paired.append(dict(case_name=case, **delta))
        comparisons[name+'_minus_'+base] = paired
    if 'bands' in records:
        prefix = p['calibration']['prefix_epochs']
        for name in records:
            if name == 'bands':
                continue
            # Every arm must share the training prefix and calibrated band strength.
            for a, b in zip(histories[name][:prefix], histories['bands'][:prefix]):
                for key in ('train_dice', 'validation_dice'):
                    if not math.isclose(float(a[key]), float(b[key]), rel_tol=1e-6, abs_tol=1e-7):
                        raise ValueError('Unmatched Dice warmup: '+name)
            a = configs[name]['calibration']['coefficients']['bands']
            b = configs['bands']['calibration']['coefficients']['bands']
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Unmatched band coefficient: '+name)
            compare(name, 'bands')
    variants = [e for e in p['runs'] if e['kind'] == 'cross' and e['id'] in records]
    for left in variants:
        for right in variants:
            keys = ('form', 'distances_mm', 'reduction', 'ap')
            changed = [k for k in keys if left[k] != right[k]]
            if len(changed) != 1:
                continue
            key = changed[0]
            preferred = dict(form='score', distances_mm=[1., 3.], reduction='anchors', ap=True)
            if right[key] != preferred[key]:
                continue
            compare(right['id'], left['id'])
            factors.append(dict(factor=key, base=left['id'], variant=right['id'],
                                macro_dice_delta=records[right['id']]['validation']['macro_dice']-
                                                 records[left['id']]['validation']['macro_dice']))
    result = dict(protocol_identity=p['identity_sha'], completed=len(rows), planned=len(p['runs']),
                  rows=rows, paired_deltas=comparisons, matched_factor_contrasts=factors,
                  interpretation='Exploratory fold 0 / seed 0; validation selected checkpoints. No confirmatory inference.')
    atomic_json(root/'results.json', result)
    write_history(root/'results.csv', rows)
    lines = ['# Ordinary SwinUNETR edge ablation: 5%, fold 0, seed 0', '',
             f"Completed: {len(rows)}/{len(p['runs'])}. Dice+bands is the fresh matched baseline.", '',
             '| Objective | A/P Dice (%) | Δ Dice (pp) | Extra | Missed | Swaps | Best epoch |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        delta = 100*(row['macro_dice']-records['bands']['validation']['macro_dice']) if 'bands' in records else float('nan')
        lines.append(f"| {row['objective']} | {100*row['macro_dice']:.4f} | {delta:+.4f} | {row['extra']} | {row['missed']} | {row['swaps']} | {row['best_epoch']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result
