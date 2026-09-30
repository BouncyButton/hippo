"""Compact causal summary; consume completed audit artifacts only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
div = read(ROOT.parent / 'boundary_diversity_20260929/RESULTS.json')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
for name, label, color in [('ten_repeated', '10 cases × 5 repeats', '#936548'), ('fifty_unique', '50 distinct cases', '#2463a6')]:
    for split, linestyle in [('train', '--'), ('validation', '-')]:
        points = [(int(e)*50, r[name][split]['mean']['macro_dice']*100) for e, r in div['fixed_epochs'].items()]
        ax[0].plot(*zip(*points), linestyle=linestyle, color=color, marker='o', markersize=4,
            label=label if split == 'validation' else None)
ax[0].set(title='More distinct cases improve generalization', xlabel='Matched optimizer updates', ylabel='Macro A/P Dice (%)')
ax[0].legend(frameon=False, fontsize=9, loc='center right')
ax[0].text(.03, .12, 'Solid: validation · dashed: training\nActual learning rates and update counts matched', transform=ax[0].transAxes, fontsize=8.5)

for seed, folder, color in [(0, 'boundary_translation_20260929', '#14856e'), (1, 'boundary_translation_replication_20260929', '#aa5c24')]:
    root = ROOT.parent / folder
    completion = root / 'results/completion.json'
    if not completion.exists():
        continue
    assert read(completion)['status'] == 'complete'
    r = read(root / 'RESULTS.json')['arms']
    values = [(r['dice']['validation'], 'identity_mean'),
              (r['dice_aug']['validation'], 'identity_mean'),
              (r['dice_aug']['validation'], 'tta_mean')]
    for axis, metric, factor in [(ax[1], 'macro_dice', 100), (ax[2], 'boundary_union_errors', 52)]:
        ys = [v[metric][key]*factor for v, key in values]
        axis.plot(range(3), ys, marker='o', lw=2, color=color, label=f'Seed {seed}')
        for x, y in enumerate(ys):
            label = f'{y:.2f}' if metric == 'macro_dice' else f'{y:,.0f}'
            # Separate the close Dice labels in these two completed runs.
            above = seed == 0
            if metric == 'macro_dice' and x > 0:
                above = seed == 1
            axis.annotate(label, (x, y), xytext=(0, 8 if above else -16), textcoords='offset points', ha='center', fontsize=9, color=color)
        axis.set_xticks(range(3), ['Dice', '+ mild\naugmentation', '+ 13-view\naverage'])
        axis.margins(x=.16, y=.28)
ax[1].set(title='Augmentation and fixed-view inference', ylabel='Validation macro A/P Dice (%)')
ax[2].set(title='Unique validation boundary errors', ylabel='Shell FP + FN, summed over 52 cases')
ax[1].legend(frameon=False, fontsize=9)
for axis in ax:
    axis.grid(axis='y', alpha=.18)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('Controlled interventions on the same development fold · selected checkpoints in right panels', fontsize=12.5)
fig.savefig(ROOT / 'causal_summary.png', dpi=180)
fig.savefig(ROOT / 'causal_summary.pdf')
plt.close(fig)
