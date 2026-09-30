"""Summarize a completed coherence audit with paired case-level uncertainty."""
from pathlib import Path
import argparse
import json
import numpy as np


KEYS = ['macro_dice', 'union_dice', 'all_fp', 'all_fn',
        'all_soft_squared_error', 'inner_soft_squared_error', 'outer_soft_squared_error',
        'cross_soft_squared_error', 'inner_disagree', 'outer_disagree',
        'inner_both_correct', 'outer_both_correct', 'inner_both_wrong', 'outer_both_wrong',
        'cross_correct_transition', 'cross_reversed_transition', 'cross_aligned_soft_jump',
        'foreground_components_6', 'foreground_outside_largest_6',
        'foreground_outside_largest_fraction_6', 'foreground_singletons_6',
        'enclosed_background_voxels_6', 'predicted_foreground_voxels']
HIGHER = {'macro_dice', 'union_dice', 'inner_both_correct', 'outer_both_correct',
          'cross_correct_transition', 'cross_aligned_soft_jump'}
METHODS = ['dice', 'bands', 'edge', 'dice_volume_matched']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = json.loads((args.directory / 'cases.json').read_text())
    complete = json.loads((args.directory / 'completion.json').read_text())
    assert complete['status'] == 'complete' and complete['model_count'] == 27
    assert len(rows) == 9 * 4 * 62
    index = {(r['split'], r['fold'], r['seed'], r['method'], r['case_name']): r['metrics'] for r in rows}
    assert len(index) == len(rows)
    names = {split: {f: sorted({r['case_name'] for r in rows if r['split'] == split and r['fold'] == f})
                     for f in range(3)} for split in ('train', 'validation')}
    summary = {'completion': complete, 'splits': {}}
    rng = np.random.default_rng(20260928)
    for split, folds in names.items():
        n = 52 if split == 'validation' else 10
        assert all(len(ids) == n for ids in folds.values())
        unique = sorted(set().union(*map(set, folds.values())))
        overlap = sum(map(len, folds.values())) - len(unique)
        result = dict(unique_case_ids=len(unique), repeated_ids_across_folds=overlap, methods={}, contrasts={}, runs=[])
        arrays = {}
        for method in METHODS:
            arrays[method] = {}
            for metric in KEYS:
                if method == 'dice_volume_matched' and metric == 'macro_dice':
                    continue
                a = np.array([[[index[split, f, s, method, name][metric] for name in folds[f]]
                              for s in range(3)] for f in range(3)], dtype=float)
                assert np.isfinite(a).all(), (method, metric)
                arrays[method][metric] = a
            result['methods'][method] = {k: float(a.mean()) for k, a in arrays[method].items()}
            for f in range(3):
                for s in range(3):
                    result['runs'].append(dict(fold=f, seed=s, method=method,
                         metrics={k: float(a[f, s].mean()) for k, a in arrays[method].items()}))
        if overlap:
            # Shared case weights preserve dependence when a case appears in
            # several folds. Fold means remain equally weighted.
            weights = rng.exponential(size=(10000, len(unique)))
            loc = {name: j for j, name in enumerate(unique)}
            fw = [weights[:, [loc[name] for name in folds[f]]] for f in range(3)]
            fw = [w / w.sum(1, keepdims=True) for w in fw]
            result['bootstrap'] = 'Shared exponential case weights across overlapping folds; seeds averaged first; equal fold means.'
        else:
            draws = [rng.integers(0, n, size=(10000, n)) for f in range(3)]
            result['bootstrap'] = 'Paired case bootstrap within each fold; seeds averaged first; equal fold means; 10000 replicates.'
        for candidate, reference in [('edge', 'dice'), ('edge', 'bands'), ('bands', 'dice'), ('edge', 'dice_volume_matched')]:
            comparison = {}
            for metric in arrays[candidate].keys() & arrays[reference].keys():
                d = arrays[candidate][metric] - arrays[reference][metric]
                seed_mean = d.mean(1)
                if overlap:
                    sampled = np.mean([fw[f] @ seed_mean[f] for f in range(3)], axis=0)
                else:
                    sampled = np.mean([seed_mean[f][draws[f]].mean(1) for f in range(3)], axis=0)
                run_delta = d.mean(2)
                comparison[metric] = dict(delta=float(d.mean()),
                    interval95=np.quantile(sampled, [.025, .975]).tolist(),
                    fold_deltas=d.mean((1, 2)).tolist(), run_deltas=run_delta.tolist(),
                    runs_improved=int(((run_delta > 0) if metric in HIGHER else (run_delta < 0)).sum()),
                    cases_improved=int(((seed_mean > 0) if metric in HIGHER else (seed_mean < 0)).sum()),
                    cases_tied=int((seed_mean == 0).sum()))
            result['contrasts'][candidate + '_vs_' + reference] = comparison
        summary['splits'][split] = result
    drift = [abs(r['metrics']['macro_dice_minus_historical_audit']) for r in rows if r['method'] != 'dice_volume_matched']
    summary['historical_dice_max_absolute_drift'] = max(drift)
    summary['historical_dice_mean_absolute_drift'] = float(np.mean(drift))
    summary['limitations'] = 'Exploratory audit of development validation-selected models. Intervals condition on these fitted models; they exclude checkpoint selection, retraining and new-fold uncertainty. Case files are the resampling units, not verified independent patients.'
    (args.directory / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'validation': summary['splits']['validation']['methods'],
                      'drift': summary['historical_dice_max_absolute_drift']}, indent=2))
    plot(summary, args.directory)
    report(summary, args.directory)


def report(summary, output):
    v = summary['splits']['validation']
    specs = [
        ('macro_dice', 'Foreground macro Dice (%)', 100),
        ('union_dice', 'Whole hippocampus Dice (%)', 100),
        ('all_fp', 'False positive voxels per case', 1),
        ('all_fn', 'False negative voxels per case', 1),
        ('outer_disagree', 'Outer same-side face disagreement (%)', 100),
        ('inner_disagree', 'Inner same-side face disagreement (%)', 100),
        ('outer_both_correct', 'Outer pairs both correct (%)', 100),
        ('inner_both_correct', 'Inner pairs both correct (%)', 100),
        ('cross_correct_transition', 'True crossing faces correctly oriented (%)', 100),
        ('all_soft_squared_error', 'Signed contrast MSE over all supported faces', 1),
        ('outer_soft_squared_error', 'Outer same-side probability MSE', 1),
        ('inner_soft_squared_error', 'Inner same-side probability MSE', 1),
        ('cross_soft_squared_error', 'True crossing signed contrast MSE', 1),
        ('foreground_components_6', 'Foreground six-connected components per case', 1),
        ('foreground_outside_largest_6', 'Foreground voxels outside largest component', 1)]
    lines = ['# Adjacent voxel coherence in the low data experiments', '',
        'Inference-only comparison of 27 selected checkpoints: three methods, three folds and three seeds. '
        'Each fold has ten training and 52 validation case files. Training had a 75-epoch maximum; these '
        'are the original validation-selected checkpoints, not necessarily the final epoch.', '',
        '## What changed', '']
    for metric, label, factor in (specs[4], specs[5], specs[8], specs[9]):
        a, b = [v['methods'][m][metric] * factor for m in ('dice', 'edge')]
        c = v['contrasts']['edge_vs_dice'][metric]
        lines.append(f'- {label}: {a:.4f} → {b:.4f}; improvement in {c["runs_improved"]}/9 fold–seed comparisons.')
    lines += ['', 'Neighbor agreement must be read together with correctness and true boundary transitions. '
        'Fewer disagreements can also arise from a uniformly incorrect region. Higher Dice alone does not imply better spatial coherence.', '',
        '## Validation means', '',
        'Each value is an equal mean of cases, seeds and folds. The FN/FP entries here are per case; multiply by 52 to compare with the summary document\'s mean counts per validation split.', '',
        '| Measurement | Baseline | Bands | Bands + edge |', '|---|---:|---:|---:|']
    for key, label, factor in specs:
        vals = [v['methods'][m][key] * factor for m in ('dice', 'bands', 'edge')]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{x:.4f}' for x in vals) + ' |')
    lines += ['', '## Paired effects and uncertainty', '',
        'Differences are candidate minus reference. Negative is better for disagreement and error; positive is better for correct transitions. Intervals average seeds before resampling cases.', '',
        '| Comparison | Measurement | Difference | Conditional 95% interval | Runs improved |',
        '|---|---|---:|---:|---:|']
    for contrast, label in [('edge_vs_dice', 'Edge vs baseline'), ('edge_vs_bands', 'Edge vs bands')]:
        for key, title, factor in (specs[4], specs[5], specs[8], specs[9]):
            c = v['contrasts'][contrast][key]
            lo, hi = np.array(c['interval95']) * factor
            lines.append(f'| {label} | {title} | {c["delta"]*factor:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {c["runs_improved"]}/9 |')
    lines += ['', '## All nine comparisons', '',
        'Entries show baseline → bands → bands + edge. Disagreement and correct-transition values are percentages.', '',
        '| Fold | Seed | Outer disagreement | Inner disagreement | Correct boundary transitions | Signed contrast MSE |',
        '|---:|---:|---|---|---|---|']
    for f in range(3):
        for s in range(3):
            cells = []
            for key, factor in [('outer_disagree', 100), ('inner_disagree', 100), ('cross_correct_transition', 100), ('all_soft_squared_error', 1)]:
                vals = [next(r['metrics'][key] for r in v['runs'] if (r['fold'],r['seed'],r['method']) == (f,s,m))*factor for m in ('dice','bands','edge')]
                cells.append(' → '.join(f'{x:.4f}' for x in vals))
            lines.append(f'| {f} | {s} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Same foreground volume diagnostic', '',
        'For each case, the baseline foreground probabilities were ranked and exactly as many voxels selected as in bands + edge. No GT was used to set the volume. This is a different decision rule, not a trained control or causal decomposition.', '',
        '| Measurement | Baseline at edge volume | Bands + edge | Difference |', '|---|---:|---:|---:|']
    for key, label, factor in (specs[1], specs[4], specs[5], specs[8], specs[13], specs[14]):
        a = v['methods']['dice_volume_matched'][key] * factor
        b = v['methods']['edge'][key] * factor
        lines.append(f'| {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines += ['', '## Scope and checks', '',
        f'- {v["unique_case_ids"]} distinct validation case IDs; {v["repeated_ids_across_folds"]} repeated occurrences across folds.',
        '- Checkpoint hashes and selected epochs matched the original audits; model-source and data hashes were verified.',
        f'- Maximum absolute macro-Dice deviation from the historical audit: {summary["historical_dice_max_absolute_drift"]:.9g}.',
        '- ' + v['bootstrap'], '- ' + summary['limitations'],
        '- Global connectivity and anatomical correctness are not guaranteed by local face metrics. Both losses supervise the foreground union and do not directly constrain A/P identity.', '',
        'Full per-case results, training measurements, all metrics, and checkpoint provenance are retained in cases.json, SUMMARY.json and bindings.json.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['dice', 'bands', 'edge']
    names = ['Baseline', '+ bands', '+ bands + edge']
    panels = [('outer_disagree', 'Outer band neighbor disagreement', 'Adjacent pairs (%)', 100),
              ('inner_disagree', 'Inner band neighbor disagreement', 'Adjacent pairs (%)', 100),
              ('cross_correct_transition', 'Correct true boundary transitions', 'Crossing pairs (%)', 100),
              ('all_soft_squared_error', 'Signed probability contrast error', 'Mean squared error', 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout='constrained')
    runs = summary['splits']['validation']['runs']
    colors = ['#19647e', '#7b4f9e', '#d57928']
    for ax, (metric, title, ylabel, factor) in zip(axes.ravel(), panels):
        for fold in range(3):
            for seed in range(3):
                values = [next(r['metrics'][metric] for r in runs if r['fold'] == fold and r['seed'] == seed and r['method'] == m) * factor for m in methods]
                ax.plot(range(3), values, color=colors[fold], alpha=.38, marker='o', lw=1)
        means = [summary['splits']['validation']['methods'][m][metric] * factor for m in methods]
        ax.plot(range(3), means, color='#182331', marker='o', lw=2.6, label='Nine-run mean')
        ax.set_xticks(range(3), names)
        ax.set(title=title, ylabel=ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        for i, v in enumerate(means):
            ax.annotate(f'{v:.2f}' if factor == 100 else f'{v:.4f}', (i, v), xytext=(5, 8), textcoords='offset points', fontsize=9)
    fig.suptitle('Adjacent voxel coherence at 5% training data\nThree folds × three seeds · selected checkpoints from the 75 epoch cap', fontsize=14)
    fig.savefig(output / 'coherence.png', dpi=180)
    fig.savefig(output / 'coherence.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
