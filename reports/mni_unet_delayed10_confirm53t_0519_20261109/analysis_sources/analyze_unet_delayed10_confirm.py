"""Report every fixed UNet confirmation pair, with pilot and new pairs separated."""
import json
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import unet_delayed10_confirm as study
from scripts.mni_constraints.analyze_unet_bands_timing_strength import load, METRICS


def summarize(rows, expected):
    gains = [r['gain_pp'] for r in rows]
    return dict(pairs=len(rows), positive_pairs=sum(g > 0 for g in gains),
                mean_gain_pp=mean(gains) if gains else None, minimum_gain_pp=min(gains, default=None),
                all_positive=len(rows) == expected and all(g > 0 for g in gains))


def analyze():
    stage, report = study.ROOT/'experiments'/study.NAME, study.ROOT/'reports'/study.NAME
    controls = {(c['condition'], c['fold'], c['seed']): c for c in read(stage/'controls.json')}
    rows = []
    for seed in study.SEEDS:
        for fold in range(3):
            baseline = load(stage/'controls'/controls['baseline', fold, seed]['id'])
            reused = (fold, seed) in study.REUSED_PAIRS
            if reused:
                folder = stage/'controls'/controls['late10', fold, seed]['id']
            else:
                folder = report/'runs'/f'MNI_unet_late10_f{fold}s{seed}'
                if not (folder/'archive_acknowledgment.json').exists():
                    continue
            run = load(folder);cfg = run['config'];cal = run['calibration']
            protocol = stage/'protocols/late10'/('parent_protocol.json' if reused else 'protocol.json')
            assert cfg['protocol_sha256'] == digest(protocol)
            assert cfg['schedule'] == read(protocol)['models']['unet']['objectives']['dice_bands']
            assert (cfg['fold'], cfg['seed']) == (fold, seed) and cfg['max_steps'] == 938
            for key in ('train_cases', 'validation_cases', 'dataset_sha256', 'initial_lr', 'augmentation',
                        'script_sha256', 'shared_source_sha256', 'objective_source_sha256'):
                assert cfg[key] == baseline['config'][key], key
            assert cal['anchor_successful_updates'] == cal['anchor_training_attempts'] == 470
            assert cal['target_ratio'] == .1 and cal['p95_cap'] == .75
            assert cal['config_sha256'] == digest(folder/'config.json')
            assert set(c['case_name'] for c in cal['cases']) == set(cfg['train_cases'])
            v, b = run['validation']['summary'], baseline['validation']['summary']
            rows.append(dict(seed=seed, fold=fold, reused_pilot=reused,
                baseline_dice=100*b['macro_hard_dice'], dice=100*v['macro_hard_dice'],
                gain_pp=100*(v['macro_hard_dice']-b['macro_hard_dice']),
                per_metric_gain_pp={k:100*(v[k]-b[k]) for k in METRICS},
                training_gain_pp={k:100*(run['training']['summary'][k]-baseline['training']['summary'][k]) for k in METRICS},
                calibration_weight=cal['weights']['bands'], calibration_statistics=cal['statistics'],
                updates=run['done']['updates'], optimization_flops=run['done']['consumed_training_flops'],
                calibration_flops=run['done']['calibration_flops'], archive=run['archive'],
                archive_sha256=run['archive_sha256'], baseline_archive=baseline['archive'],
                baseline_archive_sha256=baseline['archive_sha256'], cleaned=run['cleaned']))
    complete = len(rows) == 9 and all(r['cleaned'] for r in rows)
    result = dict(status='complete' if complete else 'running', rows=rows,
        all_pairs=summarize(rows, 9), new_confirmation_pairs=summarize([r for r in rows if not r['reused_pilot']], 7),
        pilot_pairs=summarize([r for r in rows if r['reused_pilot']], 2),
        by_seed={str(s):summarize([r for r in rows if r['seed'] == s], 3) for s in study.SEEDS},
        by_fold={str(f):summarize([r for r in rows if r['fold'] == f], 3) for f in range(3)},
        confirmation_passed=complete and all(r['gain_pp'] > 0 for r in rows),
        manifest_sha256=digest(stage/'manifest.json'), analysis_source_sha256=digest(Path(__file__)),
        caveat='Two pilot development pairs reused. Seven new confirmation pairs. Original bands/base unchanged. Same optimization cap; calibration separately accounted. Baseline939 vs bands938 updates. No automatic matrix promotion.')
    write(report/'confirmation_results.json', result)
    # Shared operational resume helper checks archive completion, never success.
    write(report/'pilot_results.json', result)
    if complete:
        lines = ['# UNet delayed 10% bands confirmation', '',
                 'Original union bands, Dice-only AdamW LR 0.00016, no augmentation. Activation at update 470, ramp 76 updates, calibrated gradient target 10%, p95 guard 75%. Optimization cap 53,305,109,592,450 FLOPs; calibration accounted separately under the established convention. All constrained runs completed 938 updates; matched archived baselines completed 939.', '',
                 '| Seed | Fold | Baseline Dice (%) | Delayed bands Dice (%) | Gain (pp) | Evidence |',
                 '|---:|---:|---:|---:|---:|---|']
        for row in rows:
            lines.append(f"| {row['seed']} | {row['fold']} | {row['baseline_dice']:.2f} | {row['dice']:.2f} | {row['gain_pp']:+.2f} | {'Reused pilot' if row['reused_pilot'] else 'New confirmation'} |")
        lines += ['', f"All nine pairs: {result['all_pairs']['positive_pairs']}/9 positive; mean gain {result['all_pairs']['mean_gain_pp']:+.3f} percentage points; minimum gain {result['all_pairs']['minimum_gain_pp']:+.3f} points.", '',
                  f"Seven new pairs: {result['new_confirmation_pairs']['positive_pairs']}/7 positive; mean gain {result['new_confirmation_pairs']['mean_gain_pp']:+.3f} points.", '',
                  'Consistency criterion (all nine paired gains positive): '+('PASSED.' if result['confirmation_passed'] else 'NOT MET.'), '',
                  'Two development pairs were reused exactly, with no additional training. Folds overlap and all results use development validation, so this is not an independent-test or significance claim. No recipe or seed was changed during confirmation; no full-matrix promotion was made.', '',
                  'All seven new runs have verified durable archives, complete Dice/FN/FP/internal-swap diagnostics, and completed checkpoint cleanup. Frozen inputs and source, calibration records, curves, configurations, logs, and archive SHA-256 receipts are preserved in this report directory.', '']
        (report/'findings.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    result = analyze()
    print(json.dumps({k:result[k] for k in ('status', 'all_pairs', 'new_confirmation_pairs', 'confirmation_passed')}))
    print(json.dumps([{k:r[k] for k in ('seed', 'fold', 'reused_pilot', 'baseline_dice', 'dice', 'gain_pp', 'cleaned')} for r in result['rows']]))
