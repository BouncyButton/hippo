"""Recompute all final tables from preserved case-level results."""
import csv
import json
from pathlib import Path
import numpy as np

OUT = Path('docs/experiments/unwrapped_followup_20261007/final_results')
EVIDENCE = OUT / 'evidence'
NEW = EVIDENCE / 'unwrapped_followup_fold0_20261007_01'
OLD = EVIDENCE / 'unwrapped_swin_fold0_seed0_20261006_02'
METRICS = ['anterior_dice', 'posterior_dice', 'foreground_dice', 'macro_dice',
           'missed', 'extra', 'swaps', 'surface_distance_mean_mm', 'hd95_mm']


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def table(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = read(NEW / 'protocol.json')
    runs, by_id, per_case, depths = [], {}, [], []

    def add(path, specification):
        completion = read(path / 'completion.json')
        result = read(path / 'validation_best.json')
        rows = sorted(result['cases'], key=lambda x: x['case_name'])
        assert [r['case_name'] for r in rows] == sorted(protocol['splits'][specification['fraction']]['val'])
        summary = {k: sum(r[k] for r in rows) if k in ('missed', 'extra', 'swaps')
                   else float(np.mean([r[k] for r in rows if r[k] is not None])) for k in METRICS}
        for k, v in summary.items():
            assert abs(v - completion['validation'][k]) < 1e-10, (specification['id'], k)
            assert abs(v - result['summary'][k]) < 1e-10
        history = list(csv.DictReader((path / 'metrics.csv').open()))
        assert len(history) == completion['completed_epochs']
        assert max(float(r['validation_dice']) for r in history) == summary['macro_dice']
        best_row = next(r for r in history if int(r['epoch']) == completion['best_epoch'])
        assert float(best_row['validation_dice']) == summary['macro_dice']
        row = dict(**specification, best_epoch=completion['best_epoch'],
                   completed_epochs=completion['completed_epochs'], **summary,
                   checkpoint_sha256=completion['artifacts']['best.pt'],
                   evidence_path=str(path.relative_to(OUT)))
        runs.append(row)
        by_id[row['id']] = {'record': row, 'cases': rows}
        for c in rows:
            per_case.append(dict(run_id=row['id'], fraction=row['fraction'], seed=row['seed'],
                                 **{k: v for k, v in c.items() if k != 'depth_bands'}))
        bands = {}
        for c in rows:
            for b in c['depth_bands']:
                aggregate = bands.setdefault(b['signed_bin'], {k: 0 for k in ('union', 'missed', 'extra', 'swaps')})
                for k in aggregate:
                    aggregate[k] += b[k]
        for band, values in sorted(bands.items()):
            depths.append(dict(run_id=row['id'], signed_bin=band, lower_mm=band*2, upper_mm=(band+1)*2,
                               **values, error_rate=sum(values[k] for k in ('missed', 'extra', 'swaps')) / values['union']))
        for k in ('missed', 'extra', 'swaps'):
            assert sum(b[k] for b in bands.values()) == summary[k]

    arms = ['dice', 'depth', 'plane_local', 'unfold_local', 'plane_context', 'unfold_context',
            'plane_position', 'unfold_position', 'plane_attention', 'unfold_attention']
    for fraction in ('5pct', 'full'):
        for arm in arms:
            add(OLD / 'runs' / fraction / arm,
                dict(id=f'prior/{fraction}/{arm}', fraction=fraction, seed=0, arm=arm,
                     objective='dice' if arm == 'dice' else 'depth', gate='none', group='original'))
    for spec in protocol['runs']:
        add(NEW / 'runs' / spec['id'], spec)
    assert len(runs) == 44 and len(by_id) == 44 and len(per_case) == 44*52
    comparisons = []

    def compare(group, reference, candidate):
        a, b = by_id[reference], by_id[candidate]
        delta = np.array([y['macro_dice'] - x['macro_dice'] for x, y in zip(a['cases'], b['cases'])]) * 100
        indices = np.random.default_rng(0).integers(52, size=(10000, 52))
        lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        comparisons.append(dict(group=group, fraction=a['record']['fraction'], seed=a['record']['seed'],
            reference=reference, candidate=candidate, dice_delta_pp=float(delta.mean()),
            case_bootstrap_low_pp=float(lo), case_bootstrap_high_pp=float(hi),
            improved_cases=int((delta > 0).sum()), worsened_cases=int((delta < 0).sum()),
            **{k+'_delta': b['record'][k] - a['record'][k] for k in ('missed','extra','swaps','surface_distance_mean_mm','hd95_mm')}))

    def key(fraction, arm, seed=0):
        return f'prior/{fraction}/{arm}' if seed == 0 else f'{fraction}_replication_s{seed}_{arm}_none'

    seeds = {}
    for fraction, selected in protocol['pairs'].items():
        for variant in ('local', 'context', 'position', 'attention'):
            compare('original_unfold_vs_plane', key(fraction, 'plane_'+variant), key(fraction, 'unfold_'+variant))
        compare('original_depth_vs_dice', key(fraction, 'dice'), key(fraction, 'depth'))
        for seed in (0, 1, 2):
            for arm in selected:
                compare('branch_vs_native', key(fraction, 'depth', seed), key(fraction, arm, seed))
            compare('selected_unfold_vs_plane', key(fraction, selected[0], seed), key(fraction, selected[1], seed))
        seeds[fraction] = {}
        for arm in ['depth', *selected]:
            records = [by_id[key(fraction, arm, s)]['record'] for s in (0, 1, 2)]
            scores = [v['macro_dice'] * 100 for v in records]
            seeds[fraction][arm] = dict(dice_pct=scores, mean_dice_pct=float(np.mean(scores)),
                sample_sd_dice_pp=float(np.std(scores, ddof=1)),
                mean_metrics={k:float(np.mean([v[k] for v in records])) for k in METRICS})
        for arm in selected:
            ablation = f'{fraction}_dice_ablation_s0_{arm}_none'
            compare('add_depth', ablation, key(fraction, arm))
            compare('dice_branch_vs_native', key(fraction, 'dice'), ablation)
            for gate in ('entropy', 'mean_entropy'):
                compare('gate_vs_ordinary', key(fraction, arm), f'{fraction}_uncertainty_s0_{arm}_{gate}')
            compare('spatial_vs_uniform', f'{fraction}_uncertainty_s0_{arm}_mean_entropy', f'{fraction}_uncertainty_s0_{arm}_entropy')
        compare('dice_only_unfold_vs_plane', f'{fraction}_dice_ablation_s0_{selected[0]}_none', f'{fraction}_dice_ablation_s0_{selected[1]}_none')
    diagnostics = []
    for fraction, selected in protocol['pairs'].items():
        for arm in selected:
            path = NEW / 'diagnostics' / fraction / arm
            completion = read(path / 'completion.json')
            audit = read(path / 'introduced_errors.json')
            rows = audit['cases']
            count = sum(r['introduced_missed'] for r in rows)
            entropy = sum(r['introduced_missed'] * (r['introduced_missed_native_entropy_mean'] or 0) for r in rows) / count
            values = {k: sum(r[k] for r in rows) for k in ('introduced_missed','recovered_missed',
                     'introduced_missed_within_2mm','introduced_missed_beyond_2mm')}
            assert values['introduced_missed'] - values['recovered_missed'] == completion['summaries']['normal']['missed'] - completion['summaries']['native_only']['missed']
            diagnostics.append(dict(fraction=fraction, arm=arm, summaries=completion['summaries'], **values,
                introduced_missed_native_entropy_voxel_mean=entropy, depth_bands=audit['depth_bands'],
                normal_dice_delta_from_saved=audit['normal_dice_delta_from_saved']))
    write('all_results.json', runs)
    write('matched_comparisons.json', comparisons)
    write('seed_summary.json', seeds)
    write('checkpoint_audits.json', diagnostics)
    table('all_results.csv', runs)
    table('per_case_metrics.csv', per_case)
    table('depth_band_metrics.csv', depths)
    table('matched_comparisons.csv', comparisons)
    write('analysis_validation.json', dict(passed=True, unique_scientific_runs=44, cases_per_run=52,
        case_rows=len(per_case), depth_band_rows=len(depths), matched_comparisons=len(comparisons), checkpoint_audits=4,
        all_saved_summary_metrics_recomputed=True, all_best_epochs_match_history=True,
        bootstrap=dict(replicates=10000, seed=0, sampling_unit='paired validation case',
                       limitation='Conditional on selected checkpoint, fixed split and geometry; not a confidence interval over pipeline seeds or independent test performance.')))
    print(json.dumps({'runs':len(runs), 'comparisons':len(comparisons), 'seed_summary':seeds, 'diagnostics':diagnostics}, indent=2))


if __name__ == '__main__':
    main()
