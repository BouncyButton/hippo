"""Plot observed baseline learning curves against the proposed lower budgets."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.cross_dataset_campaign.common import read


def main():
    root = Path(__file__).resolve().parents[2] / 'reports/msd_unet_ce_constraints_20261014'
    audit = read(root / 'lower_budget_curve_audit.json')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 5.6))
    low, high = [r['budget_flops'] / 1e12 for r in audit['candidates']]
    ax.axvspan(low, high, color='#8196a5', alpha=.16, label='Budget for 5–6 nnUNet passes')
    for key, name, color in [('nnunet', 'nnUNet', '#b35f24'), ('basic_unet16', '3D UNet', '#246697')]:
        rows = audit['curves'][key]
        x = [r['estimated_training_flops'] / 1e12 for r in rows]
        y = [100 * r['macro_hard_dice'] for r in rows]
        ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=4, label=name)
        ax.annotate(f'{y[-1]:.2f}%', (x[-1], y[-1]), xytext=(-5, 9),
                    textcoords='offset points', ha='right', color=color, fontweight='bold')
    ax.set(xlabel='Training compute consumed (TFLOPs)', ylabel='Validation Dice (%)',
           xlim=(0, 33), ylim=(0, 100))
    ax.grid(axis='y', alpha=.18)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    fig.suptitle('A lower common budget affects the models differently',
                 x=.09, ha='left', fontsize=15, fontweight='bold')
    fig.text(.09, .91, 'MSD · fold 0 · seed 0 · Dice+CE · no augmentation · identical 10/52 split')
    fig.text(.09, .025, 'Points are saved validation evaluations from the existing runs, using their original learning-rate schedules.\n'
             'The shaded interval is 17.76–21.32 TFLOPs. Curves do not establish multi-seed effects or outcomes under retuned schedules.', fontsize=8)
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.82)
    for suffix in ('png', 'pdf'):
        fig.savefig(root / f'lower_budget_tradeoff.{suffix}', dpi=180)
    print(root / 'lower_budget_tradeoff.png')


if __name__ == '__main__':
    main()
