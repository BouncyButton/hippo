"""Audit the completed fixed-update, replayed-LR diversity intervention."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / 'boundary_causal_20260929'
spec = importlib.util.spec_from_file_location('pilot_analysis', PILOT / 'analyze.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    result = {}
    for split in ('train', 'validation'):
        rr = [r for r in rows if r['split'] == split]
        assert len(rr) == len({r['case_name'] for r in rr})
        if split == 'validation':
            assert len(rr) == 52
        result[split] = dict(cases=len(rr), mean={
            k: float(np.mean([r['metrics'][k] for r in rr])) for k in pilot.METRICS
            if all(r['metrics'].get(k) is not None for r in rr)}, totals={
            k: sum(r['metrics'][k] for r in rr) for k in
            ('boundary_union_errors', 'all_fp', 'all_fn', 'ap_swaps', 'all_wrong')})
    return result


def paired_common(a, b):
    common = {(r['split'], r['case_name']) for r in a} & {(r['split'], r['case_name']) for r in b}
    return pilot.pair(*[[r for r in rows if (r['split'], r['case_name']) in common] for rows in (a, b)])


def main():
    small = ROOT / 'results/seed0/sum10_repeated'
    large = PILOT / 'results/seed0/sum'
    result = dict(models={}, fixed_epochs={}, comparisons={}, matching={},
        interpretation='The treatment changes unique case diversity and repeated exposure at fixed updates and actual learning rates. One seed and a reused validation fold; case intervals omit seed and selection uncertainty. Train totals must not be compared across different cohort sizes. Paired training scores use only the original common ten cases.')
    curves = {}
    for name, folder in [('ten_repeated', small), ('fifty_unique', large)]:
        done = read(folder / 'completion.json')
        assert done['status'] == 'complete'
        raw = folder / 'selected_cases.json'
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == done['selected_cases_sha256']
        rows = read(raw)
        result['models'][name] = dict(selected_epoch=done['selected_epoch'], stopped_epoch=done['stopped_epoch'], **summarize(rows))
        curves[name] = [read(p) for p in sorted((folder / 'epochs').glob('*.json'))]
    a, b = curves['ten_repeated'], curves['fifty_unique']
    assert len(a) == len(b) == 60
    assert [r['epoch'] for r in a] == list(range(1, 61))
    cfg = read(small / 'config.json')
    names = set(cfg['cohort']['train'])
    assert len(names) == 10
    for x, y in zip(a, b):
        for k in ('learning_rate', 'constraint_scale', 'optimizer_updates_seen'):
            assert x[k] == y[k], (x['epoch'], k)
        assert x['skipped_updates'] == y['skipped_updates'] == 0
        assert Counter(x['case_order']) == Counter({n: 5 for n in names})
        assert len(set(y['case_order'])) == len(y['case_order']) == 50
    result['matching'] = dict(passes=60, updates=3000, actual_learning_rates_identical=True,
        constraint_warmup_identical=True, amp_skips=0, ten_cases_each_repeated_five_times_each_pass=True)
    result['comparisons']['selected_fifty_minus_ten'] = paired_common(read(small / 'selected_cases.json'), read(large / 'selected_cases.json'))
    for epoch in (15, 30, 45, 60):
        x, y = [read(f / f'audits/epoch_{epoch:03d}.json') for f in (small, large)]
        result['fixed_epochs'][str(epoch)] = dict(ten_repeated=summarize(x), fifty_unique=summarize(y), fifty_minus_ten=paired_common(x, y))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Training diversity with matched updates and actual learning rates', '', result['interpretation'], '',
        '| Model | Selected/stop | Train cases | Train Dice% | Validation Dice% | Shell errors, 52 validation cases |',
        '|---|---:|---:|---:|---:|---:|']
    for name, r in result['models'].items():
        lines.append(f"| {name} | {r['selected_epoch']}/{r['stopped_epoch']} | {r['train']['cases']} | {100*r['train']['mean']['macro_dice']:.3f} | {100*r['validation']['mean']['macro_dice']:.3f} | {r['validation']['totals']['boundary_union_errors']} |")
    lines += ['', 'All 60 passes match actual learning rates, warmup and updates; no AMP steps were skipped.', '', 'Fifty minus ten, validation effects:', '']
    for label, comparisons in [('selected', result['comparisons']['selected_fifty_minus_ten'])] + [(f'pass {e}', r['fifty_minus_ten']) for e, r in result['fixed_epochs'].items()]:
        v = comparisons['validation']
        lines.append(f"- {label}: Dice {100*v['macro_dice']['mean_change']:+.3f} percentage points; shell errors {52*v['boundary_union_errors']['mean_change']:+.0f}; ASSD {v['assd_voxels']['mean_change']:+.4f} grid voxels.")
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
