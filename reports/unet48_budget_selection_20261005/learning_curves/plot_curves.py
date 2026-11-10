"""Plot verified metrics from independent width-48 baseline schedules."""
from pathlib import Path
import csv
import hashlib
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
archive = root / 'completed_learning_curves.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'c340465242561fa9797d2acdd64891666932a69b8bd91ebbb40dc6741835d0a0'
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        relative = Path(member.name)
        assert member.isfile() and len(relative.parts) == 1 and relative.suffix == '.csv'
        (root / relative).write_bytes(tar.extractfile(member).read())

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
for candidate, cap, color in zip([6, 8, 7, 5, 4], [4000, 4200, 4400, 4800, 6400], colors):
    rows = list(csv.DictReader((root / f'candidate{candidate:02d}.csv').open()))
    updates = [int(r['step']) for r in rows]
    axes[0].plot(updates, [100 * float(r['macro_hard_dice']) for r in rows],
                 marker='.', linewidth=1.5, color=color, label=f'{cap:,}-update schedule')
    axes[1].plot(updates, [100 * float(r['posterior_soft_dice']) for r in rows],
                 marker='.', linewidth=1.5, color=color)
    axes[2].plot(updates, [float(r['learning_rate']) for r in rows],
                 linewidth=1.5, color=color)

axes[0].axhspan(70, 75, color='#999999', alpha=.16, label='Selection target')
axes[0].set(title='Validation macro hard Dice', ylabel='Dice (%)', ylim=(0, 100))
axes[1].set(title='Validation posterior soft Dice', ylabel='Dice (%)', ylim=(0, 100))
axes[2].set(title='Learning rate at each update', ylabel='AdamW learning rate', yscale='log', ylim=(8e-7, 1.2e-4))
for ax in axes:
    ax.set_xlabel('Successful training updates')
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(fontsize=8, loc='lower right')
fig.suptitle('U-Net width 48 · independent Dice-only runs · fold 0, seed 0', fontsize=12)
fig.savefig(root / 'budget_learning_curves.png', dpi=180)
fig.savefig(root / 'budget_learning_curves.pdf')
print(root / 'budget_learning_curves.png')
