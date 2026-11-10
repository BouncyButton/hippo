"""Compare verified SAM3 archives and diagnose the constraint gradients."""
import argparse
import math
from pathlib import Path
from statistics import mean, median

from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.cross_dataset_campaign.sam3_four_volume_job import NAME, ROOT
from scripts.cross_dataset_campaign.storage import verify_archive


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    return dict(count=len(values), median=median(values), p95=quantile(values, .95),
                minimum=min(values), maximum=max(values)) if values else None


def gradient_diagnostics(calibration):
    """Use saved training-only Gram matrices; no fresh model evaluations."""
    if calibration['status'] == 'not_required':
        return dict(auxiliaries={}, combined=None)
    names, weights = calibration['names'], calibration['weights']
    coefficients = [1.] + [weights[name] for name in names[1:]]
    auxiliaries, combined_ratios, combined_cosines = {}, [], []
    for i, name in enumerate(names[1:], 1):
        native_cosines, sequential_cosines, native_ratios, sequential_ratios = [], [], [], []
        for row in calibration['rows']:
            g = row['gram']
            previous_norm2 = sum(coefficients[j] * coefficients[k] * g[j][k]
                                 for j in range(i) for k in range(i))
            if g[0][0] > 0 and g[i][i] > 0:
                native_cosines.append(g[0][i] / math.sqrt(g[0][0] * g[i][i]))
                native_ratios.append(weights[name] * math.sqrt(g[i][i] / g[0][0]))
            if previous_norm2 > 0 and g[i][i] > 0:
                sequential_cosines.append(sum(coefficients[j] * g[j][i] for j in range(i))
                                          / math.sqrt(previous_norm2 * g[i][i]))
                sequential_ratios.append(weights[name] * math.sqrt(g[i][i] / previous_norm2))
        saved = calibration['statistics'][name]
        for actual, expected in ((median(sequential_ratios), saved['weighted_median']),
                                 (quantile(sequential_ratios, .95), saved['weighted_p95'])):
            if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError('Reconstructed calibration ratio differs')
        auxiliaries[name] = dict(weight=weights[name],
            sequential_gradient_ratio=distribution(sequential_ratios),
            native_gradient_ratio=distribution(native_ratios),
            cosine_with_native=distribution(native_cosines),
            cosine_with_preceding_objective=distribution(sequential_cosines),
            fraction_opposed_to_native=mean(v < 0 for v in native_cosines),
            p95_ceiling_binding=math.isclose(saved['weighted_p95'], calibration['p95_cap'], rel_tol=1e-6))
    for row in calibration['rows']:
        g = row['gram']
        aux_norm2 = sum(coefficients[j] * coefficients[k] * g[j][k]
                        for j in range(1, len(names)) for k in range(1, len(names)))
        if aux_norm2 > 0 and g[0][0] > 0:
            combined_ratios.append(math.sqrt(aux_norm2 / g[0][0]))
            combined_cosines.append(sum(coefficients[j] * g[0][j] for j in range(1, len(names)))
                                   / math.sqrt(aux_norm2 * g[0][0]))
    return dict(auxiliaries=auxiliaries, combined=dict(
        native_gradient_ratio=distribution(combined_ratios), cosine_with_native=distribution(combined_cosines)),
        interpretation='Initial training-only gradients at full constraint weight, before warmup and clipping; not measured throughout training.')


def report(destination):
    frozen = destination/'frozen'
    plan = read(frozen/'plan.json')
    for name, sha in read(frozen/'manifest.json').items():
        if digest(frozen/name) != sha:
            raise ValueError('Frozen report source changed: '+name)
    rows, case_maps = [], {}
    for task in plan['tasks']:
        run = destination/'runs'/task['id']
        if not (run/'archive_acknowledgment.json').exists():
            continue
        manifest = read(run/'archive_manifest.json')
        verify_archive(run, manifest)
        ack = read(run/'archive_acknowledgment.json')
        if (manifest['task'] != task or ack['manifest_sha256'] != digest(run/'archive_manifest.json')
                or ack['source_manifest_sha256'] != digest(frozen/'manifest.json')):
            raise ValueError('Archive provenance mismatch')
        done, evaluation, history, calibration, protocol = (
            read(run/(name+'.json')) for name in
            ('completion', 'evaluation', 'training_history', 'calibration', 'protocol'))
        if done['updates'] != 200 or len(history) != 200 or done['primary_checkpoint'] != 'final_update':
            raise ValueError('Incomplete final-update evaluation')
        if task['arm'] != 'dice' and (calibration['target'], calibration['p95_cap']) != (task['target'], task['p95_cap']):
            raise ValueError('Wrong gradient strength')
        records = {r['case_name']: r for r in evaluation['records'] if r['split'] == 'validation'}
        expected = read(frozen/'protocols'/task['protocol']/'protocol.json')
        if (set(records) != set(expected['evaluation_cases']) or len(records) != 96
                or protocol['train_cases'] != expected['train_cases']
                or protocol['base_sha256'] != expected['base_sha256']):
            raise ValueError('Evaluation or support cohort changed')
        case_maps[task['id']] = records
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        rows.append(dict(**task, validation_dice=done['validation_dice'], training_dice=done['training_dice'],
            foreground_fn=sum(r['foreground_fn'] for r in records.values()),
            foreground_fp=sum(r['foreground_fp'] for r in records.values()),
            gradients=gradient_diagnostics(calibration),
            clipping_fraction=mean(h['gradient_norm'] > 1. for h in history),
            pre_clip_norm=distribution([h['gradient_norm'] for h in history]),
            base_sha256=protocol['base_sha256'], checkpoint_sha256=done['checkpoint_sha256'],
            cleanup_status=cleanup.get('status', 'pending'), released_bytes=cleanup.get('released_file_bytes', 0)))
    baseline = next((r for r in rows if r['arm'] == 'dice'), None)
    if baseline:
        reference = case_maps[baseline['id']]
        for row in rows:
            cases = case_maps[row['id']]
            if any(cases[k]['truth_sha256'] != reference[k]['truth_sha256'] for k in reference):
                raise ValueError('Validation annotations differ between conditions')
            changes = [dict(case=k, delta_dice_pp=100*(cases[k]['macro_hard_dice']-reference[k]['macro_hard_dice']))
                       for k in sorted(reference)]
            row.update(delta_dice_pp=100*(row['validation_dice']-baseline['validation_dice']),
                       delta_fn=row['foreground_fn']-baseline['foreground_fn'],
                       delta_fp=row['foreground_fp']-baseline['foreground_fp'], paired_changes=changes,
                       improved_volumes=sum(c['delta_dice_pp'] > 0 for c in changes),
                       worsened_volumes=sum(c['delta_dice_pp'] < 0 for c in changes))
    result = dict(status='complete' if len(rows) == 7 else 'incomplete', completed=len(rows), total=7,
        methodology='Matched four-volume SAM3 comparisons on 96 validation volumes; 200 updates, fold0/seed0, no augmentation. Targets refer to sequential LoRA gradient ratios against the native SAM3 objective, which includes Dice. This is exploratory single-seed evidence.',
        baseline_id=baseline['id'] if baseline else None, runs=rows)
    write(destination/'comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=ROOT/'reports'/NAME)
    result = report(parser.parse_args().destination)
    print({k: result[k] for k in ('status', 'completed', 'total')})
