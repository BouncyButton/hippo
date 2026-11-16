"""Reproduce the HFH saved-run diagnostic tables and learning-curve figure."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hippo_hfh_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
records, checks = [], []
curves = {}
for folder in sorted((ROOT / 'runs').iterdir()):
    config = json.loads((folder / 'config.json').read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    metrics = list(csv.DictReader((folder / 'metrics.csv').open()))
    curves[folder.name] = metrics
    assert completion['status'] == 'complete'
    assert completion['discarded_attempts'] == 0
    for filename, expected in json.loads((folder / 'archive_manifest.json').read_text()).items():
        path = folder / filename
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            checks.append(str(path.relative_to(ROOT)))
    row = dict(run=folder.name, model=config['model_identity'], seed=config['seed'],
               fold=config['fold'], updates=completion['updates'],
               lr_horizon=config['lr_horizon_updates'],
               final_lr=float(metrics[-1]['learning_rate']),
               training_flops=completion['consumed_training_flops'],
               best_validation_dice=completion['best_validation_hard_dice'])
    for split, filename in [('train', 'training_final.json'), ('validation', 'validation_latest.json')]:
        data = json.loads((folder / filename).read_text())
        cases = [case['per_class']['hippocampus'] for case in data['cases']]
        tp, fp, fn = [sum(case[key] for case in cases) for key in ('tp', 'fp', 'fn')]
        row.update({f'{split}_dice': data['summary']['macro_hard_dice'],
                    f'{split}_tp': tp, f'{split}_fp': fp, f'{split}_fn': fn,
                    f'{split}_precision': tp / (tp + fp) if tp + fp else None,
                    f'{split}_recall': tp / (tp + fn),
                    f'{split}_predicted_to_true_volume': (tp + fp) / (tp + fn),
                    f'{split}_empty_predictions': sum(case['tp'] + case['fp'] == 0 for case in cases)})
        for case in data['cases']:
            p = case['per_class']['hippocampus']
            assert abs(2 * p['tp'] / (2 * p['tp'] + p['fp'] + p['fn']) - p['dice']) < 1e-7
        assert abs(np.mean([c['dice'] for c in cases]) - data['summary']['macro_hard_dice']) < 1e-7
    row.update(json.loads((folder / 'worker_timing.json').read_text()))
    records.append(row)

with (ROOT / 'run_summary.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(ROOT / 'analysis.json').write_text(json.dumps(dict(
    runs=records, verified_manifest_files=len(checks), verified_files=checks,
    scope='11 completed runs, fold 0 only; per-crop mean Dice, pooled voxel precision/recall',
    limitations=['No new GPU experiment or saved-checkpoint inference performed.',
                 'Archived prediction volumes and checkpoints were deleted after evaluation.',
                 'Loss-component gradients and probability histograms were not recorded.',
                 'calibration.json was not downloaded; these Dice-only runs required no calibration.',
                 'Causal effects of LR, loss, precision, padding, and training duration need controlled probes.'],
    conclusions=[
        'All 11 workers completed their prescribed optimization FLOPs allowance without discarded attempts.',
        'nnU-Net seeds 0 and 3 exhibit absent or near-absent hard foreground on train and validation.',
        'nnU-Net seeds 1 and 2 begin recovering near the cap; seed 4 recovers earlier and passes.',
        'UNETR seed 4 oversegments on train and validation and improves until its final evaluation.',
        'BasicUNet passes every attempted seed but produces substantial excess foreground.',
        'All LR horizons are roughly twice the executed updates, inherited intentionally from the frozen recipe.',
        'Controller stopped on its Drive headroom guard before seed 5; search_status is stale after rejection/crash.'
    ]), indent=2) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout='constrained')
for ax, model, title in zip(axes, ['basic_unet16', 'nnunet', 'unetr_vit_b16'],
                            ['BasicUNet · 563 updates', 'nnU-Net · 90 updates', 'UNETR · 245 updates']):
    for row in records:
        if row['model'] != model:
            continue
        data = curves[row['run']]
        ax.plot([float(x['compute_fraction']) * 100 for x in data],
                [float(x['macro_hard_dice']) * 100 for x in data],
                label=f"Seed {row['seed']}", color=plt.cm.tab10(row['seed']), linewidth=2)
    ax.axhline(50, color='#333333', linestyle='--', linewidth=1)
    ax.set(title=title, xlabel='Training FLOPs budget consumed (%)', xlim=(0, 102), ylim=(0, 80))
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
axes[0].set_ylabel('Validation Dice (%)')
fig.suptitle('HFH fold 0: recorded learning curves · dashed line = seed acceptance threshold')
fig.savefig(ROOT / 'learning_curves.png', dpi=180)
fig.savefig(ROOT / 'learning_curves.pdf')
print(json.dumps({'runs': len(records), 'verified_manifest_files': len(checks),
                  'table': str(ROOT / 'run_summary.csv'), 'figure': str(ROOT / 'learning_curves.png')}))
