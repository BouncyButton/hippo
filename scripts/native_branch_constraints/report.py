"""Matched per-case and aggregate comparisons after each completed run."""
import json
from scripts.depth_supervision.utils import atomic_json, write_history
from .train import completed_record, identity


def report(p, root):
    records, cases, rows = {}, {}, []
    for entry in p['runs']:
        folder = root/'runs'/entry['id']
        record = completed_record(folder, identity(p, entry))
        if record is None:
            continue
        objective = entry['objective']
        records[objective] = record
        cases[objective] = {r['case_name']: r for r in json.loads((folder/'validation_best.json').read_text())['cases']}
        rows.append(dict(objective=objective, best_epoch=record['best_epoch'],
                         completed_epochs=record['completed_epochs'],
                         training_seconds=record['training_seconds'], **record['validation']))
    comparisons = {}
    for objective in records:
        bases = ['dice'] if objective != 'dice' else []
        if objective.endswith('_depth') and objective.removesuffix('_depth') != 'dice':
            bases.append(objective.removesuffix('_depth'))
        for base in bases:
            if base not in records:
                continue
            assert set(cases[objective]) == set(cases[base])
            paired = []
            for name, row in cases[objective].items():
                other = cases[base][name]
                delta = {key: row[key]-other[key] for key in records[objective]['validation']
                         if key in row and key in other and isinstance(row[key], (int, float)) and isinstance(other[key], (int, float))}
                paired.append(dict(case_name=name, **delta))
            comparisons[objective+'_minus_'+base] = paired
    result = dict(protocol_identity=p['identity_sha'], completed=len(rows), planned=12,
                  rows=rows, paired_deltas=comparisons)
    atomic_json(root/'results.json', result)
    write_history(root/'results.csv', rows)
    lines = ['# Native MRI branch: 5%, fold 0, seed 0', '',
             '| Objective | A/P Dice (%) | Δ Dice (pp) | Extra | Missed | Swaps | Best epoch |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        delta = 100*(row['macro_dice']-records['dice']['validation']['macro_dice']) if 'dice' in records else float('nan')
        lines.append(f"| {row['objective']} | {100*row['macro_dice']:.4f} | {delta:+.4f} | {row['extra']} | {row['missed']} | {row['swaps']} | {row['best_epoch']} |")
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result
