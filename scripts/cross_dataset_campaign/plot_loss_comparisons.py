"""Plot the four MSD loss comparisons against the consumed FLOP budget."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]/'reports'
    current = root/'msd_other_models_loss_pairs_20261014'
    nnunet = root/'msd_nnunet_loss_pair_90updates_20261014'
    experiments = [
        ('nnUNet', nnunet, 'MSD_nnunet_native_dice_90updates_f0s0', 'MSD_nnunet_native_dice_ce_90updates_f0s0'),
        ('SwinUNETR', current, 'MSD_swinunetr_dice_f0s0', 'MSD_swinunetr_dice_ce_f0s0'),
        ('UNETR', current, 'MSD_unetr_vit_b16_dice_f0s0', 'MSD_unetr_vit_b16_dice_ce_f0s0'),
        ('3D UNet', current, 'MSD_basic_unet16_dice_f0s0', 'MSD_basic_unet16_dice_ce_f0s0'),
    ]
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.1), sharex=True, sharey=True)
    for ax, (title, report, dice, ce) in zip(axes.flat, experiments):
        for name, label, color in [(dice, 'Dice only', '#345e89'), (ce, 'Dice + CE', '#c4503e')]:
            run = report/'runs'/name
            config = json.loads((run/'config.json').read_text())
            with (run/'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            x = [100*float(row['total_flops'])/config['budget_flops'] for row in rows]
            y = [100*float(row['macro_hard_dice']) for row in rows]
            ax.plot(x, y, color=color, label=label, linewidth=2)
            ax.scatter(x[-1], y[-1], color=color, s=22, zorder=3)
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xlim(0, 103); ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.18)
    axes[0, 0].legend(loc='upper left', frameon=False)
    for ax in axes[:, 0]:ax.set_ylabel('Validation Dice (%)')
    for ax in axes[-1]:ax.set_xlabel('Training FLOP budget consumed (%)')
    fig.suptitle('MSD: effect of adding cross-entropy', fontsize=16, fontweight='bold', x=.08, ha='left')
    fig.text(.08, .925, 'Fold 0 · seed 0 · 10 training / 52 validation volumes · no augmentation', fontsize=10)
    fig.text(.08, .02, 'Each pair preserves its Dice term, initialization and optimizer. nnUNet uses native foreground Dice; the other models use their original MONAI Dice.', fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, bottom=.105, top=.85, hspace=.23, wspace=.13)
    fig.savefig(current/'loss_comparison_curves.png', dpi=180)
    fig.savefig(current/'loss_comparison_curves.pdf')
    print(current/'loss_comparison_curves.png')


if __name__ == '__main__':
    main()
