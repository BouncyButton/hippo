"""Build matched, archive-verified SAM3 stopping-point comparisons."""
from pathlib import Path
from statistics import mean, median

from scripts.cross_dataset_campaign.common import digest, read, write
from scripts.cross_dataset_campaign.report_sam3_four_volume import gradient_diagnostics, quantile
from scripts.cross_dataset_campaign.sam3_five_volume_updates_job import NAME, ROOT
from scripts.cross_dataset_campaign.storage import verify_archive


def report(destination=ROOT/'reports'/NAME):
    frozen = destination/'frozen'
    plan = read(frozen/'plan.json')
    for name, sha in read(frozen/'manifest.json').items():
        if digest(frozen/name) != sha:
            raise ValueError('Frozen campaign source changed: '+name)
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
        done, evaluation, history, calibration, protocol, training = (
            read(run/(name+'.json')) for name in
            ('completion', 'evaluation', 'training_history', 'calibration', 'protocol', 'training_summary'))
        expected = read(frozen/'protocols'/task['protocol']/'protocol.json')
        if (protocol != {**expected, 'arm':task['arm']} or done['updates'] != task['updates']
                or len(history) != task['updates'] or training['updates'] != task['updates']
                or done['primary_checkpoint'] != 'final_update'
                or training['learning_rate_horizon_updates'] != 200):
            raise ValueError('Incomplete or unmatched stopping-point result')
        records = {r['case_name']:r for r in evaluation['records'] if r['split'] == 'validation'}
        if set(records) != set(expected['evaluation_cases']) or len(records) != 96:
            raise ValueError('Validation cohort differs')
        if calibration['status'] == 'complete':
            for name, stats in calibration['statistics'].items():
                if stats['target'] != expected['gradient_ratio_targets'][name] or stats['p95_cap'] != expected['gradient_ratio_p95_caps'][name]:
                    raise ValueError('Auxiliary calibration differs')
        case_maps[task['id']] = records
        cleanup = read(run/'cleanup.json') if (run/'cleanup.json').exists() else {}
        gradients = gradient_diagnostics(calibration)
        for name, item in gradients['auxiliaries'].items():
            item['p95_ceiling_binding'] = abs(item['sequential_gradient_ratio']['p95'] -
                                                calibration['statistics'][name]['p95_cap']) < 1e-6
        rows.append(dict(**task, validation_dice=done['validation_dice'],
            training_dice=done['training_dice'],
            foreground_fn=sum(r['foreground_fn'] for r in records.values()),
            foreground_fp=sum(r['foreground_fp'] for r in records.values()),
            gradients=gradients, clipping_fraction=mean(h['gradient_norm'] > 1 for h in history),
            base_sha256=protocol['base_sha256'], checkpoint_sha256=done['checkpoint_sha256'],
            cleanup_status=cleanup.get('status', 'pending'),
            released_bytes=cleanup.get('released_file_bytes', 0)))
    by_key = {(r['updates'], r['arm'], r['bands_target'], r['edge_target']):r for r in rows}
    def paired(row, control):
        cases, reference = case_maps[row['id']], case_maps[control['id']]
        if any(cases[k]['truth_sha256'] != reference[k]['truth_sha256'] for k in reference):
            raise ValueError('Validation annotations differ')
        changes = [100*(cases[k]['macro_hard_dice']-reference[k]['macro_hard_dice']) for k in sorted(reference)]
        return dict(control_id=control['id'], delta_mean_dice_pp=100*(row['validation_dice']-control['validation_dice']),
                    median_case_delta_pp=median(changes), p10_case_delta_pp=quantile(changes, .1),
                    worst_case_delta_pp=min(changes), improved_volumes=sum(x > 0 for x in changes),
                    worsened_volumes=sum(x < 0 for x in changes),
                    delta_fn=row['foreground_fn']-control['foreground_fn'],
                    delta_fp=row['foreground_fp']-control['foreground_fp'],
                    per_case_delta_pp=dict(zip(sorted(reference), changes)))
    for row in rows:
        native = by_key.get((row['updates'], 'dice', .2, None))
        bands = by_key.get((row['updates'], 'dice_bands', .2, None))
        row['versus_native'] = paired(row, native) if native else None
        row['versus_bands20'] = paired(row, bands) if bands and row['arm'] == 'dice_bands_edge' else None
    result = dict(status='complete' if len(rows) == plan['num_runs'] and all(r['cleanup_status']=='complete' for r in rows) else 'partial',
                  completed=len(rows), total=plan['num_runs'], runs=rows,
                  methodology='Five frozen GeoFWI3D support volumes, 96 validation volumes, fold0/seed0, '
                              'no augmentation, fixed 200-update learning-rate horizon; final-update Dice. '
                              'Small edge targets are relative to SAM3 native-loss gradients. Exploratory screen.')
    write(destination/'comparison.json', result)
    return result


if __name__ == '__main__':
    result = report()
    print({key:result[key] for key in ('status', 'completed', 'total')})
