"""Reproduce the requested six-cell MNI comparison from verified archives."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
source = ROOT / 'final_results.json'
data = json.loads(source.read_text())
models = ['nnunet', 'swinunetr', 'unetr_vit_b16', 'basic_unet16']
labels = ['nnU-Net', 'SwinUNETR', 'UNETR', '3D U-Net']
cells = [(seed, fold) for seed in [0, 5] for fold in range(3)]
rows = [r for r in data['rows'] if r['seed'] in [0, 5]]
assert len(rows) == 4 * 14 * 6
assert len({(r['model'], r['arm'], r['seed'], r['fold']) for r in rows}) == len(rows)
for row in rows:
    validation = Path(row['archive']) / 'validation_latest.json'
    raw = validation.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['validation_sha256'], str(validation)
    assert abs(json.loads(raw)['summary']['macro_hard_dice'] - row['final_dice']) < 1e-12

fig, axes = plt.subplots(2, 2, figsize=(19, 13.5))
fig.patch.set_facecolor('#f7f9fc')
summaries = []
for ax, model, label in zip(axes.flat, models, labels):
    subset = [r for r in rows if r['model'] == model]
    lookup = {(r['arm'], r['seed'], r['fold']): r['final_dice'] * 100 for r in subset}
    values = np.array([[lookup[(arm, seed, fold)] for seed, fold in cells] for arm in data['arms']])
    means = values.mean(axis=1)
    display = np.column_stack([values, means])
    winner = int(np.argmax(means))
    best_run = max(subset, key=lambda r: r['final_dice'])
    arm_labels = ['Baseline' if a == 'dice' else a.removeprefix('dice_').replace('_', ' + ').replace('sdf', 'SDF') for a in data['arms']]
    im = ax.imshow(display, cmap='YlGnBu', vmin=55, vmax=80, aspect='auto')
    ax.set_xticks(range(7), ['s0 / f0', 's0 / f1', 's0 / f2', 's5 / f0', 's5 / f1', 's5 / f2', 'MEAN'])
    ax.xaxis.tick_top()
    ax.set_yticks(range(14), arm_labels)
    ax.tick_params(axis='both', length=0, labelsize=10, pad=7)
    ax.set_title(f'{label}   |   best mean {means[winner]:.2f}%   |   gain +{means[winner] - means[0]:.2f} pp', loc='left', fontsize=13, fontweight='bold', pad=34)
    for i in range(14):
        for j in range(7):
            ax.text(j, i, f'{display[i,j]:.2f}', ha='center', va='center', fontsize=9.5,
                    color='white' if display[i,j] > 72.5 else '#162a3b',
                    fontweight='bold' if j == 6 or i == winner else 'normal')
    ax.add_patch(Rectangle((-0.49, winner-0.48), 6.98, .96, fill=False, edgecolor='#f39722', linewidth=2.4))
    ax.axvline(5.5, color='white', linewidth=3)
    ax.axvline(2.5, color='white', linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    summaries.append({'model': label, 'baseline_mean_pct': float(means[0]),
                      'best_arm': data['arms'][winner], 'best_mean_pct': float(means[winner]),
                      'gain_pp': float(means[winner] - means[0]),
                      'best_individual': {k: best_run[k] for k in ['arm', 'seed', 'fold', 'final_dice']},
                      'baseline_cells_pct': values[0].tolist(), 'best_arm_cells_pct': values[winner].tolist()})

fig.suptitle('MNI • seeds 0 and 5 • all three folds', x=.025, y=.98, ha='left', fontsize=23, fontweight='bold', color='#142b45')
fig.text(.025, .945, 'Final-budget validation Dice (%) • mean of three foreground subfields • orange outline = best six-run average', fontsize=13, color='#38516b')
fig.text(.025, .025, 'Same condition is averaged across all six runs. Best means are selected on validation data; they are descriptive, not held-out test estimates.\nLatest completed 53.3T optimization-FLOP campaign. Calibration cost is tracked separately.', fontsize=11, color='#38516b', linespacing=1.6)
fig.subplots_adjust(left=.18, right=.98, top=.87, bottom=.10, wspace=.60, hspace=.32)
fig.savefig(ROOT / 'seeds_0_5_comparison.png', dpi=170, facecolor=fig.get_facecolor())
fig.savefig(ROOT / 'seeds_0_5_comparison.pdf', facecolor=fig.get_facecolor())
report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'verified_archive_count': len(rows), 'metric': data['metric'], 'seeds': [0, 5], 'folds': [0, 1, 2],
          'cell_order': cells, 'summaries': summaries, 'rows': rows}
(ROOT / 'seeds_0_5_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
for s in summaries:
    print(s['model'], 'baseline', round(s['baseline_mean_pct'], 2), 'best', s['best_arm'], round(s['best_mean_pct'], 2), 'gain', round(s['gain_pp'], 2))
    for i, seed in enumerate([0, 5]):
        for kind in ['baseline', 'best_arm']:
            v = s[kind + '_cells_pct'][i*3:i*3+3]
            print(seed, kind, ' | '.join(f'{x:.2f}' for x in v), 'mean', f'{statistics.mean(v):.2f}')
print('Verified all', len(rows), 'archived validation files and scores.')
