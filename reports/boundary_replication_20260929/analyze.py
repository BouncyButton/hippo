"""Keep initialization replications separate; never pool repeated subjects."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'boundary_causal_20260929'
spec = importlib.util.spec_from_file_location('pilot_analysis', PILOT / 'analyze.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
read = pilot.read


def main():
    result = dict(seeds={}, equal_seed_mean={},
        scope='Two initializations on the same 50 training and 52 validation cases. Descriptive case intervals condition on selected checkpoints and omit training and selection uncertainty. Do not treat 104 scores as 104 independent validation cases.')
    for seed, root in [(0, PILOT / 'results'), (1, ROOT / 'results')]:
        s = dict(models={}, matching={}, selected={}, fixed_epochs={})
        rows, curves, configs = {}, {}, {}
        for arm in ('dice', 'dice_aug'):
            f = root / f'seed{seed}' / arm
            done = read(f / 'completion.json')
            assert done['status'] == 'complete'
            assert hashlib.sha256((f / 'selected_cases.json').read_bytes()).hexdigest() == done['selected_cases_sha256']
            rows[arm] = read(f / 'selected_cases.json')
            configs[arm] = read(f / 'config.json')
            assert configs[arm]['run']['seed'] == seed
            s['models'][arm] = dict(selected_epoch=done['selected_epoch'], stopped_epoch=done['stopped_epoch'], **pilot.summarize(rows[arm]))
            curves[arm] = [read(p) for p in sorted((f / 'epochs').glob('*.json'))]
            assert all(r['skipped_updates'] == 0 for r in curves[arm])
            assert all(len(r['case_order']) == len(set(r['case_order'])) == 50 for r in curves[arm])
        assert configs['dice']['cohort'] == configs['dice_aug']['cohort']
        assert configs['dice']['pkl_sha256'] == configs['dice_aug']['pkl_sha256']
        common = min(map(len, curves.values()))
        assert all(a['case_order'] == b['case_order'] and a['constraint_scale'] == b['constraint_scale']
            for a, b in zip(curves['dice'], curves['dice_aug']))
        first_lr_difference = next((a['epoch'] for a, b in zip(curves['dice'], curves['dice_aug']) if a['learning_rate'] != b['learning_rate']), None)
        s['matching'] = dict(common_passes_identical_case_order=common, amp_skips=0,
            first_actual_lr_difference_pass=first_lr_difference, shared_lr_policy=True)
        s['selected'] = pilot.pair(rows['dice'], rows['dice_aug'])
        for epoch in (30, 60):
            a, b = [read(root / f'seed{seed}' / arm / f'audits/epoch_{epoch:03d}.json') for arm in ('dice', 'dice_aug')]
            s['fixed_epochs'][str(epoch)] = dict(models={k: pilot.summarize(v) for k, v in zip(('dice', 'dice_aug'), (a, b))}, augmentation_minus_control=pilot.pair(a, b))
        result['seeds'][str(seed)] = s
    for k in pilot.METRICS:
        values = [s['selected']['validation'][k]['mean_change'] for s in result['seeds'].values()]
        result['equal_seed_mean'][k] = dict(mean_change=float(np.mean(values)), seed_changes=values)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Augmentation replication', '', result['scope'], '',
        '| Seed | Dice control% | Dice augmented% | Change pp | Control shell errors | Augmented shell errors | Change | Cases with fewer errors |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed, s in result['seeds'].items():
        a, b = [s['models'][arm]['validation'] for arm in ('dice', 'dice_aug')]
        d = s['selected']['validation']
        lines.append(f"| {seed} | {100*a['mean']['macro_dice']:.4f} | {100*b['mean']['macro_dice']:.4f} | {100*d['macro_dice']['mean_change']:+.4f} | {a['totals']['boundary_union_errors']} | {b['totals']['boundary_union_errors']} | {b['totals']['boundary_union_errors']-a['totals']['boundary_union_errors']:+d} | {d['boundary_union_errors']['cases_decrease']}/52 |")
    d = result['equal_seed_mean']
    lines += ['', f"Equal-seed mean gain: {100*d['macro_dice']['mean_change']:.4f} Dice points and {-52*d['boundary_union_errors']['mean_change']:.1f} fewer shell errors per 52-case evaluation.", '', 'Fixed-budget checks:', '']
    for seed, s in result['seeds'].items():
        for epoch, r in s['fixed_epochs'].items():
            d = r['augmentation_minus_control']['validation']
            lines.append(f"- Seed{seed}, pass{epoch}: {100*d['macro_dice']['mean_change']:+.4f} Dice points; {52*d['boundary_union_errors']['mean_change']:+.0f} shell errors.")
        lines.append(f"- Seed{seed}: matching case order for {s['matching']['common_passes_identical_case_order']} passes, zero AMP skips; actual LR first differs at pass {s['matching']['first_actual_lr_difference_pass']} under the shared adaptive policy.")
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
