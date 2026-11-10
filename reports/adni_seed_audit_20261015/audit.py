"""Read-only audit of the captured Colab baseline results; no training."""
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.cross_dataset_campaign.storage import verify_archive

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent / 'adni_colab_3fold_3seed_noaug_20261015/bundle'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


report = read(ROOT / 'comparison_snapshot.json')
rows = [r for r in report['runs'] if r['arm'] == 'dice']
plan = read(BUNDLE / 'plan.json')
tasks = {t['id']: t for t in plan['tasks']}
assert len({r['id'] for r in rows}) == len(rows)
finite(report)
missing = [t['id'] for t in plan['tasks'] if t['arm'] == 'dice'
           and t['id'] not in {r['id'] for r in rows}]
details = []
for folder in sorted((ROOT / 'runs').iterdir()):
    task = tasks[folder.name]
    manifest = read(folder / 'archive_manifest.json')
    ack = read(folder / 'archive_acknowledgment.json')
    cleanup = read(folder / 'cleanup.json')
    assert manifest['task'] == task
    assert ack['status'] == 'verified' and ack['source_verified']
    assert ack['manifest_sha256'] == sha(folder / 'archive_manifest.json')
    assert ack['source_manifest_sha256'] == sha(BUNDLE / 'bundle_manifest.json')
    assert cleanup['status'] == 'complete'
    verify_archive(folder, manifest)
    for p in folder.glob('*.json'):
        finite(read(p))
    config = read(folder / 'config.json')
    done = read(folder / 'completion.json')
    ledger = read(folder / 'budget_ledger.json')
    init = read(folder / 'initialization.json')
    recipe = read(folder / 'runtime_recipe.json')
    proto_dir = BUNDLE / 'protocols' / task['protocol']
    protocol = read(proto_dir / 'protocol.json')
    split = read(proto_dir / 'splits_10patients.json')[task['fold']]
    assert config['protocol_sha256'] == sha(proto_dir / 'protocol.json')
    assert init['model_state_sha256'] == protocol['baseline_reference']['initialization_sha256']
    assert config['augmentation'] == recipe['augmentation'] == 'none'
    assert config['train_cases'] == split['train']
    assert config['validation_cases'] == split['val']
    assert config['dataset_sha256'] == protocol['pkl_sha256']
    assert config['fold'] == done['fold'] == task['fold']
    assert config['seed'] == done['seed'] == task['seed']
    assert done['status'] == 'complete' and done['discarded_attempts'] == 0
    assert done['updates'] == config['max_steps'] == ledger['charged_attempts']
    assert done['consumed_training_flops'] == ledger['training_flops']
    assert 0 <= done['consumed_training_flops'] <= protocol['budget_flops']
    assert done['budget_flops'] == config['budget_flops'] == protocol['budget_flops']
    for name, field in [('config.json', 'config_sha256'),
                        ('metrics.csv', 'metrics_sha256'),
                        ('budget_ledger.json', 'budget_ledger_sha256'),
                        ('validation_best.json', 'validation_best_sha256'),
                        ('calibration.json', 'calibration_sha256')]:
        assert done[field] == sha(folder / name)
    evaluation = read(folder / 'validation_latest.json')
    training = read(folder / 'training_final.json')
    assert len(evaluation['cases']) == 54 and len(training['cases']) == 10
    assert {c['case_name'] for c in evaluation['cases']} == set(split['val'])
    assert {c['case_name'] for c in training['cases']} == set(split['train'])
    row = next(r for r in rows if r['id'] == folder.name)
    assert row['dice'] == 100 * evaluation['summary']['macro_hard_dice']
    assert row['training_dice'] == 100 * done['final_training_hard_dice']
    curve = list(csv.DictReader((folder / 'metrics.csv').open()))
    assert int(curve[-1]['step']) == done['updates']
    assert all(math.isfinite(float(v)) for r in curve for v in r.values())
    confusion = [[sum(c['confusion'][i][j] for c in evaluation['cases'])
                  for j in range(2)] for i in range(2)]
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    details.append(dict(id=folder.name, status='archive_and_recipe_checks_passed',
        files_verified=len(manifest['metadata']), updates=done['updates'],
        budget_percent=100 * ledger['training_flops'] / protocol['budget_flops'],
        discarded_attempts=done['discarded_attempts'],
        validation_dice=row['dice'], training_dice=row['training_dice'],
        validation_precision=tp / (tp + fp), validation_recall=tp / (tp + fn),
        validation_fp=fp, validation_fn=fn,
        first_validation_dice=100 * float(curve[0]['macro_hard_dice']),
        last_validation_dice=100 * float(curve[-1]['macro_hard_dice']),
        last_curve_loss=float(curve[-1]['mean_train_loss'])))

audit = dict(completed_report_runs=report['completed'], baseline_count=len(rows),
             expected_baselines=36, missing_baselines=missing, detailed_checks=details,
             scope='All available baseline summary rows; full metadata archive verification for five weak runs. No training or result changes.')
(ROOT / 'audit_results.json').write_text(json.dumps(audit, indent=2) + '\n')
with (ROOT / 'baselines.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(audit, indent=2))
print('Model | Fold | Seed 0 | Seed 1 | Seed 2')
for model in plan['models']:
    for fold in plan['folds']:
        vals = [next((f"{r['dice']:.2f}" for r in rows
                      if (r['model'], r['fold'], r['seed']) == (model, fold, seed)), 'pending')
                for seed in plan['seeds']]
        print(model, fold, *vals, sep=' | ')
