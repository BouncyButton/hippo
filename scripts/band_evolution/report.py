"""Keep all outcomes and paired comparisons, verifying the common prefix."""
import csv
import json
import math
from scripts.depth_supervision.utils import atomic_json, write_history
from .train import completed_record, identity


def report(p, root):
    rows, cases, configs, histories = [], {}, {}, {}
    for entry in p['runs']:
        folder = root/'runs'/entry['id']
        record = completed_record(folder, identity(p, entry))
        if record is None:
            continue
        name = entry['id']
        configs[name] = json.loads((folder/'config.json').read_text())
        with (folder/'metrics.csv').open() as stream:
            histories[name] = list(csv.DictReader(stream))
        cases[name] = {r['case_name']: r for r in json.loads((folder/'validation_best.json').read_text())['cases']}
        rows.append(dict(objective=name, family=entry['family'], best_epoch=record['best_epoch'],
                         completed_epochs=record['completed_epochs'], training_seconds=record['training_seconds'],
                         **record['validation']))
    contrasts = {}
    for entry in p['runs']:
        name = entry['id']
        if name not in cases:
            continue
        if 'bands' in histories:
            for a, b in zip(histories[name][:5], histories['bands'][:5]):
                for key in ('train_dice', 'validation_dice'):
                    if float(a[key]) != float(b[key]):
                        raise ValueError('Shared warmup mismatch: '+name)
            if entry['band']['kind'] == 'canonical':
                a = configs[name]['calibration']['coefficients']['bands']
                b = configs['bands']['calibration']['coefficients']['bands']
                if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError('Canonical band calibration mismatch: '+name)
        for reference in dict.fromkeys(['bands', 'independent13', entry['compare']]):
            if reference == name or reference not in cases:
                continue
            if set(cases[name]) != set(cases[reference]):
                raise ValueError('Validation cases changed')
            paired = []
            for case, value in cases[name].items():
                other = cases[reference][case]
                delta = {k: v-other[k] for k, v in value.items() if isinstance(v, (int, float))
                         and isinstance(other.get(k), (int, float))}
                paired.append(dict(case_name=case, **delta))
            contrasts[name+'_minus_'+reference] = paired
    result = dict(protocol_identity=p['identity_sha'], completed=len(rows), planned=len(p['runs']),
                  rows=rows, paired_deltas=contrasts, interpretation=p['interpretation'])
    atomic_json(root/'results.json', result)
    write_history(root/'results.csv', rows)
    baseline = next((r['macro_dice'] for r in rows if r['objective'] == 'bands'), None)
    lines = ['# Independent targets and bands follow-up: 5%, fold 0, seed 0', '', f'Completed: {len(rows)}/{len(p["runs"])}.', '',
             '| Arm | A/P Dice % | Delta pp | Extra | Missed | Surface mm | HD95 mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        delta = 100*(row['macro_dice']-baseline) if baseline is not None else float('nan')
        lines.append(f"| {row['objective']} | {100*row['macro_dice']:.4f} | {delta:+.4f} | {row['extra']} | {row['missed']} | {row.get('surface_distance_mean_mm', float('nan')):.4f} | {row.get('hd95_mm', float('nan')):.4f} |")
    lines += ['', p['interpretation'], '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    return result
