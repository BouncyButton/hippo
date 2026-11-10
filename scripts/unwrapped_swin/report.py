"""Paired patient and voxel comparisons; run automatically in the allocation."""
import json
from pathlib import Path
import numpy as np
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
from scripts.depth_supervision.train import Patients, GeometryCache
from scripts.depth_supervision.utils import atomic_json


def report(p, root):
    comparisons = [('dice', 'depth')]
    comparisons += [('plane_' + k, 'unfold_' + k) for k in ('local', 'context', 'position', 'attention')]
    comparisons += [(v + '_' + a, v + '_' + b) for v in ('plane', 'unfold')
                    for a, b in [('local', 'context'), ('context', 'position'), ('position', 'attention')]]
    patients = Patients(_load_pkl_dataframe(p['pkl']), p['splits']['full']['val'], p['native_geometry'])
    geometry = GeometryCache(root / 'loss_geometry', p['geometry'])
    results = {}
    for fraction in ('5pct', 'full'):
        runs, paired = {}, []
        for arm in p['arms']:
            path = root / 'runs' / fraction / arm
            if (path / 'completion.json').exists():
                runs[arm] = json.loads((path / 'completion.json').read_text())
        for reference, candidate in comparisons:
            if reference not in runs or candidate not in runs:
                continue
            paths = [root / 'runs' / fraction / arm for arm in (reference, candidate)]
            metrics = [json.loads((path / 'validation_best.json').read_text()) for path in paths]
            by_name = [{r['case_name']: r for r in m['cases']} for m in metrics]
            delta = [by_name[1][n]['macro_dice'] - by_name[0][n]['macro_dice'] for n in p['splits'][fraction]['val']]
            rng = np.random.default_rng(0)
            boots = np.asarray(delta)[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
            entry = dict(reference=reference, candidate=candidate, dice_delta_mean=float(np.mean(delta)),
                         patient_bootstrap95=np.quantile(boots, [.025, .975]).tolist(),
                         patients_improved=int((np.array(delta) > 0).sum()), patients_worse=int((np.array(delta) < 0).sum()),
                         depth_bands={}, corrected=0, regressed=0, persistent_errors=0,
                         corrected_missed=0, corrected_extra=0, corrected_swaps=0,
                         created_missed=0, created_extra=0, created_swaps=0)
            with np.load(paths[0] / 'validation_predictions.npz') as old, np.load(paths[1] / 'validation_predictions.npz') as new:
                for item in patients.items:
                    name = item['case_name']; target = item['label'][0].numpy()
                    a, b = old[name], new[name]
                    error_a, error_b = a != target, b != target
                    corrected, regressed = error_a & ~error_b, ~error_a & error_b
                    entry['corrected'] += int(corrected.sum()); entry['regressed'] += int(regressed.sum())
                    entry['persistent_errors'] += int((error_a & error_b).sum())
                    for prefix, pred, mask in [('corrected', a, corrected), ('created', b, regressed)]:
                        entry[prefix + '_missed'] += int((mask & (target > 0) & (pred == 0)).sum())
                        entry[prefix + '_extra'] += int((mask & (target == 0) & (pred > 0)).sum())
                        entry[prefix + '_swaps'] += int((mask & (target > 0) & (pred > 0)).sum())
                    bins = geometry.get(item, fields=False)['bin_ids']
                    union = (target > 0) | (a > 0) | (b > 0)
                    for k in np.unique(bins[union]):
                        mask = (bins == k) & union
                        row = entry['depth_bands'].setdefault(str(int(k)), dict(union=0, corrected=0, regressed=0, old_errors=0, new_errors=0))
                        for key, value in [('union', mask), ('corrected', mask & corrected), ('regressed', mask & regressed),
                                           ('old_errors', mask & error_a), ('new_errors', mask & error_b)]:
                            row[key] += int(value.sum())
            paired.append(entry)
        results[fraction] = dict(runs=runs, comparisons=paired)
    atomic_json(root / 'comparison.json', dict(protocol_sha256=p['identity_sha'], fractions=results,
                uncertainty='Patient bootstrap intervals are descriptive; one training seed and repeated validation selection do not establish generalization.'))
    return results
