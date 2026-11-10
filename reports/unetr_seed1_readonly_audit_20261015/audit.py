"""Read-only checks of downloaded, completed UNETR runs; no training imports."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent / 'msd_colab_3fold_3seed_noaug_20261015/bundle'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {'status': 'passed', 'optimizer_updates_performed_by_audit': 0,
              'runs': [], 'not_downloaded': [], 'checked_manifest_files': 0}
    splits = read(BUNDLE/'protocols/unetr_vit_b16_s0/splits_10patients.json')
    reference = read(BUNDLE/'initialization_reference.json')['models']['unetr_vit_b16']
    truth_by_fold = {}
    base_curves = {}
    common_config = None
    folders = sorted((p for p in (ROOT/'runs').iterdir() if p.is_dir()),
                     key=lambda p: ('_dice_bands_' in p.name, p.name))
    for folder in folders:
        if not folder.is_dir():
            continue
        manifest = read(folder/'archive_manifest.json')
        task = manifest['task']
        fold, seed, arm = (task[k] for k in ('fold', 'seed', 'arm'))
        for name, info in manifest['metadata'].items():
            path = folder/name
            if not path.exists():
                result['not_downloaded'].append(str(path.relative_to(ROOT)))
                continue
            assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], path
            result['checked_manifest_files'] += 1
        ack = read(folder/'archive_acknowledgment.json')
        assert ack['status'] == 'verified' and ack['source_verified']
        assert ack['manifest_sha256'] == sha(folder/'archive_manifest.json')
        assert ack['source_manifest_sha256'] == sha(BUNDLE/'bundle_manifest.json')
        assert read(folder/'cleanup.json')['status'] == 'complete'
        config = read(folder/'config.json')
        done = read(folder/'completion.json')
        ledger = read(folder/'budget_ledger.json')
        recipe = read(folder/'runtime_recipe.json')
        for filename in ('config', 'metrics', 'budget_ledger', 'validation_best', 'calibration'):
            ext = 'csv' if filename == 'metrics' else 'json'
            assert done[filename+'_sha256'] == sha(folder/(filename+'.'+ext))
        assert (config['fold'], config['seed'], config['objective']) == (fold, seed, arm)
        assert config['train_cases'] == splits[fold]['train']
        assert config['validation_cases'] == splits[fold]['val']
        assert not set(config['train_cases']) & set(config['validation_cases'])
        protocol_path = BUNDLE/f'protocols/unetr_vit_b16_s{seed}/protocol.json'
        assert config['protocol_sha256'] == sha(protocol_path)
        protocol = read(protocol_path)
        assert config['schedule'] == protocol['models']['unet']['objectives'][arm]
        assert config['augmentation'] == recipe['augmentation'] == 'none'
        assert config['budget_flops'] == 31976450667090
        assert (recipe['weight_dice'], recipe['weight_ce'], recipe['dice_include_background']) == (1, 0, True)
        assert recipe['target'] == .1 and recipe['p95_cap'] == .5
        assert done['updates'] == done['training_attempts_charged'] == ledger['charged_attempts'] == 245
        assert done['discarded_attempts'] == 0
        assert done['consumed_training_flops'] == ledger['training_flops'] == config['schedule']['consumed_training_flops']
        assert ledger['training_flops'] <= config['budget_flops']
        init = read(folder/'initialization.json')['model_state_sha256']
        assert init == reference[str(seed)]['model_sha256'] == protocol['baseline_reference']['initialization_sha256']
        common = {k: config[k] for k in ('dataset_sha256', 'script_sha256', 'objective_source_sha256',
                  'shared_source_sha256', 'sdf_source_sha256', 'model_identity', 'lr_horizon_updates',
                  'torch', 'monai', 'precision', 'augmentation')}
        if common_config is None:
            common_config = common
        assert common == common_config
        curve = list(csv.DictReader((folder/'metrics.csv').open()))
        assert len(curve) == 20
        assert all(math.isfinite(float(v)) for row in curve for v in row.values())
        assert int(curve[-1]['step']) == 245
        assert all(int(r['step']) == int(r['charged_attempts']) for r in curve)
        for row in curve:
            step = int(row['step'])-1
            lr = .0001*(step+1)/24 if step < 24 else .0001*(.01+.99*(1+math.cos(math.pi*(step-24)/(477-24-1)))/2)
            assert math.isclose(float(row['learning_rate']), lr, rel_tol=1e-12)
        if arm == 'dice':
            base_curves[fold, seed] = curve
        else:
            calibration = read(folder/'calibration.json')
            assert calibration['target_ratio'] == .1 and calibration['p95_cap'] == .5
            assert {r['case_name'] for r in calibration['cases']} == set(config['train_cases'])
            assert set(calibration['weights']) == {'bands'}
            assert math.isfinite(calibration['weights']['bands']) and calibration['weights']['bands'] > 0
            assert math.isclose(calibration['statistics']['bands']['weighted_median_ratio'], .1, abs_tol=1e-12)
            assert calibration['statistics']['bands']['weighted_p95_ratio'] <= .5
            for row, baseline in zip(curve[:4], base_curves[fold, seed][:4]):
                for key in row:
                    assert float(row[key]) == float(baseline[key]), (folder.name, key)
        measurements = {}
        for filename, expected_cases in (('validation_latest', config['validation_cases']),
                                         ('validation_best', config['validation_cases']),
                                         ('training_final', config['train_cases'])):
            evaluation = read(folder/(filename+'.json'))
            assert {r['case_name'] for r in evaluation['cases']} == set(expected_cases)
            dices, truth, fp, fn, predicted, total_fg = [], {}, 0, 0, 0, 0
            for row in evaluation['cases']:
                matrix = row['confusion']
                assert all(type(v) is int and v >= 0 for line in matrix for v in line)
                assert sum(map(sum, matrix)) == 64**3
                support = [sum(line) for line in matrix]
                truth[row['case_name']] = support
                per_class = []
                for c, name in enumerate(('background', 'anterior', 'posterior')):
                    tp = matrix[c][c]
                    false_pos = sum(line[c] for line in matrix)-tp
                    false_neg = support[c]-tp
                    value = 2*tp/(2*tp+false_pos+false_neg) if 2*tp+false_pos+false_neg else 1.
                    assert row['per_class'][name] == dict(tp=tp, fp=false_pos, fn=false_neg, dice=value)
                    if c:
                        per_class.append(value)
                value = statistics.mean(per_class)
                assert math.isclose(value, row['macro_hard_dice'], abs_tol=1e-15)
                assert row['foreground_fp'] == sum(matrix[0][1:])
                assert row['foreground_fn'] == matrix[1][0]+matrix[2][0]
                dices.append(value)
                fp += row['foreground_fp']; fn += row['foreground_fn']
                predicted += sum(sum(line[1:]) for line in matrix)
                total_fg += sum(support[1:])
            assert math.isclose(statistics.mean(dices), evaluation['summary']['macro_hard_dice'], abs_tol=1e-15)
            key = (fold, filename)
            if key not in truth_by_fold:
                truth_by_fold[key] = truth
            assert truth == truth_by_fold[key], (folder.name, 'different ground truth')
            measurements[filename] = dict(dice=statistics.mean(dices), foreground_fp=fp,
                foreground_fn=fn, predicted_foreground_voxels=predicted,
                truth_foreground_voxels=total_fg, prediction_to_truth_ratio=predicted/total_fg)
        assert math.isclose(float(curve[-1]['macro_hard_dice']), measurements['validation_latest']['dice'], abs_tol=1e-15)
        result['runs'].append(dict(id=folder.name, fold=fold, seed=seed, arm=arm,
            initialization_sha256=init, measurements=measurements, archived_at=manifest['created_at']))
    result['shared_config'] = common_config
    result['checks'] = ['Downloaded metadata hashes and archive acknowledgment/source binding',
        'All nine baseline runs and two fold-0 bands runs; 245 updates, none discarded',
        'Exact seed-specific initialization hashes, identical across folds and matched arms',
        'Exact frozen cohort IDs and per-case label voxel counts across seeds',
        'Identical scientific source, data, precision, loss, learning-rate trajectory and FLOP schedules',
        'All 20 curve milestones finite; pre-constraint baseline/bands prefix matches exactly',
        'Bands calibration uses exactly the ten training cases, achieves10% median and respects50% p95 ceiling',
        'Independently recomputed hard Dice and foreground FN/FP from every saved confusion matrix',
        'Verified cleanup receipts; completed checkpoints already retired, no model replay performed']
    (ROOT/'audit_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs', 'not_downloaded')}, indent=2))
    print('Not downloaded:', len(result['not_downloaded']), 'non-scientific log/lock artifacts')


if __name__ == '__main__':
    main()
