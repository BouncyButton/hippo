"""Write the paper-facing spatial error summary after every model is verified."""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
NAMES = {
    'basic_unet16': '3D U-Net',
    'nnunet': 'nnU-Net',
    'unetr_vit_b16': 'UNETR',
    'swinunetr': 'Swin UNETR',
    'medsam3': 'MedSAM3',
}
DATASETS = ('MSD', 'MNI', 'ADNI', 'HFH')
BAND = 'boundary_fg_bg_share_all_errors_pct'
EDGE = 'same_side_disagreement_pct'
ISOLATED = 'crops_with_isolated_error_pct'
LOW = 'lower_population_boundary_error_rate_pct'
HIGH = 'higher_population_boundary_error_rate_pct'
FAR = 'class_errors_beyond_2_pct'


def write_coverage(cells):
    """Show historical coverage alongside checks that have all three partitions."""
    with (HERE / 'original_baselines_20_cells.csv').open() as source:
        original = {(r['model'], r['dataset']): r for r in csv.DictReader(source)}
    early, late = cells.xs(5, level='step'), cells.xs(50, level='step')
    lines = [
        '# Baseline coverage and spatial error measurements', '',
        f'The original paper supplies nine controls for each of 20 settings. '
        f'The separate spatial analysis has verified {int(late.partitions.sum())} of 60 diagnostic runs. '
        'Spatial rates appear once all three partitions are complete for that model and dataset. '
        'Each diagnostic uses seed 1741 and measures all held-out crops at updates 5 and 50.', '',
        '| Model | Dataset | Original runs | Foreground/background, % of original label errors | '
        'Diagnostic partitions | Band errors, % of all | Discordant same-side faces, % | '
        'Crops with isolated error, % | Lower / higher layer error rates, % | Distant class errors, update 5 → 50, % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for model, name in NAMES.items():
        for dataset in DATASETS:
            key = model, dataset
            r = original[key]
            n = int(late.loc[key, 'partitions']) if key in late.index else 0
            line = f'| {name} | {dataset} | {r["runs"]} | {float(r["external_error_pct"]):.1f} | {n}/3 |'
            if n != 3:
                lines.append(line + ' — | — | — | — | — |')
                continue
            a, b = early.loc[key], late.loc[key]
            lines.append(line + f' {b[BAND]:.1f} | {b[EDGE]:.1f} | {b[ISOLATED]:.1f} | '
                         f'{b[LOW]:.1f} / {b[HIGH]:.1f} | {a[FAR]:.1f} → {b[FAR]:.1f} |')
    lines.extend(['', 'A foreground/background confusion can occur anywhere in the crop. The band '
                  'column counts only these confusions within the two-step outer hippocampal band. '
                  'Internal foreground-class swaps remain in its denominator. For edge consistency '
                  'the denominator is all same-side neighboring faces inside the band. Depth rates '
                  'divide band errors by the available band voxels in each layer group.'])
    dest = HERE / 'coverage.md'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(dest)


def write_report(cells, runs):
    early = cells.xs(5, level='step')
    late = cells.xs(50, level='step')
    expected = {(model, dataset) for model in NAMES for dataset in DATASETS}
    assert set(early.index) == set(late.index) == expected
    assert (cells.partitions == 3).all()
    assert runs.groupby('step').crop_evaluations.sum().to_dict() == {5: 1876, 50: 1876}
    early_mean = early.mean()
    late_mean = late.mean()
    fewer_distant = int((early[FAR] > late[FAR]).sum())
    sparse_higher = int((late[LOW] > late[HIGH]).sum())
    no_overlap = int(((runs.step == 50) & (runs.mean_crop_foreground_dice_pct == 0)).sum())
    paragraph = (
        'We examined held-out baseline predictions from five models on four datasets '
        '(three partitions per setting, one initialization per partition). '
        f'At update 50, {late_mean[BAND]:.1f}% of label errors were foreground/background '
        'confusions within two voxel steps of the outer contour, motivating boundary-band supervision. '
        f'Within these bands, foreground predictions disagreed across {late_mean[EDGE]:.1f}% of '
        'face-sharing pairs whose reference labels were on the same side of the contour, motivating '
        f'edge consistency. Boundary error rates averaged {late_mean[LOW]:.1f}% in less populated '
        f'depth layers and {late_mean[HIGH]:.1f}% in more populated layers, motivating separate '
        'supervision by layer. Finally, the share of classwise errors more than two coordinate units '
        f'from their reference boundary was {early_mean[FAR]:.1f}% at update 5 and '
        f'{late_mean[FAR]:.1f}% at update 50, motivating distance-sensitive supervision.'
    )
    lines = [
        '# Spatial errors in baseline predictions',
        '',
        paragraph,
        '',
        '## Coverage and variation',
        '',
        'These are descriptive measurements from 60 supplementary baseline runs using seed 1741. '
        'Each update has 1,876 held-out crop evaluations. The same crops are observed again at the '
        'second update. The rates pool voxel counts within each partition, then weight the three '
        'partitions, five models and four datasets equally. Updates represent different amounts of '
        'computation across architectures.',
        '',
        f'At update 50, isolated foreground/background errors occurred in {late_mean[ISOLATED]:.1f}% '
        'of crops on average. An isolated error has six observable face neighbors whose '
        'foreground/background predictions are all correct. '
        f'The less populated layers had the higher boundary error rate in {sparse_higher} of '
        f'20 model–dataset combinations. The fraction of distant class errors fell in '
        f'{fewer_distant} of 20 combinations. At update 50, {no_overlap} of the 60 runs had zero '
        'mean foreground Dice. A spatially constant prediction can give zero same-side disagreement '
        'while missing an entire structure. These measurements characterize baseline errors. '
        'Testing whether signed-distance supervision accelerates learning requires a matched '
        'comparison with signed-distance training.',
        '',
        '| Model | Dataset | Mean crop Dice at update 50, % | Band errors, % of all | Discordant same-side faces, % | '
        'Crops with isolated error, % | Lower / higher layer error rates, % | '
        'Distant class errors, update 5 → 50, % |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for model, name in NAMES.items():
        for dataset in DATASETS:
            a, b = early.loc[(model, dataset)], late.loc[(model, dataset)]
            lines.append(
                f'| {name} | {dataset} | {b["mean_crop_foreground_dice_pct"]:.2f} | '
                f'{b[BAND]:.2f} | {b[EDGE]:.2f} | {b[ISOLATED]:.2f} '
                f'| {b[LOW]:.2f} / {b[HIGH]:.2f} | {a[FAR]:.2f} → {b[FAR]:.2f} |'
            )
    dest = HERE / 'spatial_results.md'
    tmp = dest.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(dest)
