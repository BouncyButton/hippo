"""Verify the four high-target runs and compare all five target levels."""
import argparse
from pathlib import Path

from scripts.cross_dataset_campaign.common import read, write
from scripts.cross_dataset_campaign.report_unet_constraints import report


def build(root):
    previous = report(root.parent/'msd_unet_ce_constraints_20261014')
    current = report(root)
    for key in ('dataset', 'model', 'fold', 'seed', 'augmentation',
                'baseline_dice_ce_percent', 'baseline_updates', 'budget_flops'):
        if previous[key] != current[key]:
            raise ValueError('The high-target comparison differs from the previous pilot: '+key)
    rows = []
    for experiment, path in ((previous, root.parent/'msd_unet_ce_constraints_20261014'), (current, root)):
        for row in experiment['runs']:
            rows.append(dict(row, archive=str(path/'runs'/row['id'])))
    rows.sort(key=lambda r: (r['objective'], r['target']))
    baseline = read(root.parent/'msd_other_models_loss_pairs_20261014/runs/MSD_basic_unet16_dice_ce_f0s0/validation_latest.json')
    baseline_errors = {k: sum(c[k] for c in baseline['cases']) for k in ('foreground_fp', 'foreground_fn')}
    baseline_errors.update({k: sum(c['ap_swaps'][k] for c in baseline['cases'])
                            for k in ('anterior_to_posterior', 'posterior_to_anterior')})
    for row in rows:
        row['voxel_error_change_vs_baseline'] = {k: row['voxel_errors'][k]-v for k, v in baseline_errors.items()}
    result = dict(current, runs=rows, new_run_count=len(current['runs']),
                  baseline_voxel_errors=baseline_errors,
                  scope='Ten matched runs: bands and bands+original edge+depth CE at 10/40/60/80/100% targets.',
                  high_target_p95_ceilings={'0.8': 1., '1.0': 1.25},
                  ratio_interpretation='Each auxiliary is calibrated sequentially against the current joint gradient. The percentage is not a fixed multiplier on the scalar Dice loss.')
    write(root/'all_targets_comparison.json', result)
    return result


def plot(root, result):
    if result['status'] != 'complete' or len(result['runs']) != 10:
        raise ValueError('All ten archived runs are required for the complete plot')
    from scripts.cross_dataset_campaign.plot_unet_constraints import curve
    import matplotlib.pyplot as plt
    baseline = root.parent/'msd_other_models_loss_pairs_20261014/runs/MSD_basic_unet16_dice_ce_f0s0'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    colors = {0.1: '#4278a5', 0.4: '#d38425', 0.6: '#85529e', 0.8: '#1d8b72', 1.: '#b64253'}
    for ax, arm, title in zip(axes, ('dice_bands', 'dice_bands_edge_depth'),
                             ('Bands', 'Bands + original edge + depth CE')):
        ax.plot(*curve(baseline), color='#333333', linestyle='--', linewidth=2, label='Dice+CE baseline: 81.18%')
        for row in result['runs']:
            if row['objective'] != arm:
                continue
            x, y = curve(Path(row['archive']))
            ax.plot(x, y, color=colors[row['target']], linewidth=1.8,
                    label=f'{int(100*row["target"])}% target: {y[-1]:.2f}% Dice')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set(xlim=(0, 101), ylim=(0, 100), xlabel='Training FLOP budget consumed (%)')
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, loc='lower right', fontsize=9)
    axes[0].set_ylabel('Validation Dice (%)')
    fig.suptitle('3D UNet: extending constraint targets to 80% and 100%', x=.07, ha='left', fontsize=16, fontweight='bold')
    fig.text(.07, .91, 'MSD · fold 0 · seed 0 · Dice+CE · 10 training / 52 validation volumes · no augmentation')
    fig.text(.07, .025, 'Final-budget comparison. High-target p95 ceilings: 100% / 125%. Gradient calibration is sequential and uses training volumes only.', fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, bottom=.14, top=.81, wspace=.13)
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'all_targets_comparison.{suffix}', dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('reports/msd_unet_ce_constraints_high_20261014'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    result = build(args.root)
    if args.plot:
        plot(args.root, result)
    for row in result['runs']:
        print(f'{row["objective"]} {100*row["target"]:.0f}%: '
              f'{row["validation_dice_percent"]:.4f}% ({row["gain_vs_dice_ce_baseline_pp"]:+.4f} pp), cleanup {row["cleanup"]}')


if __name__ == '__main__':
    main()
