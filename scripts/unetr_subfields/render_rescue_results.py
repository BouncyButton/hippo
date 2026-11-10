"""Export the verified 36-run comparison as CSV and a scientific figure."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.cross_dataset_campaign.common import read, write
from scripts.unetr_subfields.contracts import ROOT


def main():
    destination = ROOT / 'reports/mni_baseline_rescue_20261106'
    result = read(destination / 'final_results.json')
    assert result['status'] == 'complete' and result['completed_runs'] == 36
    labels = dict(nnunet='nnUNet', swinunetr='SwinUNETR',
                  unetr_vit_b16='UNETR · Geo recipe', basic_unet16='U-Net')
    records = []
    recipes = {}
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.3), layout='constrained')
    for ax, summary in zip(axes.flat, result['models']):
        assert summary['target_met'] and len(summary['runs']) == 9
        recipes[summary['model']] = dict(
            stage=summary['name'], source_manifest_sha256=summary['source_manifest_sha256'],
            protocol=read(ROOT / 'reports' / summary['name'] / 'frozen/protocols/model/protocol.json'))
        matrix = np.full((3, 3), np.nan)
        for row in summary['runs']:
            assert np.isnan(matrix[row['seed'], row['fold']])
            matrix[row['seed'], row['fold']] = row['validation_dice'] * 100
            records.append(dict(model=row['model'], fold=row['fold'], seed=row['seed'],
                                validation_macro_dice=row['validation_dice'],
                                training_macro_dice=row['training_dice'],
                                whole_hippocampus_dice=row['whole_hippocampus_dice'],
                                **{key + '_dice': value for key, value in row['subfields'].items()},
                                updates=row['updates'], budget_flops=result['budget_flops']))
        assert np.isfinite(matrix).all()
        im = ax.imshow(matrix, vmin=45, vmax=85, cmap='YlGnBu')
        for seed in range(3):
            for fold in range(3):
                value = matrix[seed, fold]
                ax.text(fold, seed, f'{value:.1f}', ha='center', va='center',
                        color='white' if value >= 67 else '#142536', fontsize=16)
        ax.set(xticks=range(3), yticks=range(3), xlabel='Fold', ylabel='Seed',
               title=f"{labels[summary['model']]}\nMean {matrix.mean():.1f}% · lowest run {matrix.min():.1f}%")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=.8, label='Final-budget validation foreground macro Dice (%)')
    fig.suptitle(f"MNI · 3 folds × 3 fixed seeds\nShared optimization cap: {result['historical_cap_multiplier']:g}× historical cap", fontsize=15)
    fig.supxlabel('Five training subjects per fold. Development validation informed recipe selection.', fontsize=9)
    fig.savefig(destination / 'final_dice_matrix.png', dpi=180)
    fig.savefig(destination / 'final_dice_matrix.pdf')
    plt.close(fig)
    with (destination / 'final_runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write(destination / 'final_recipes.json', recipes)
    print('Exported 36 verified runs and four-model Dice matrix.')


if __name__ == '__main__':
    main()
