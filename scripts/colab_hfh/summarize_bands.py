"""Verify downloaded HFH BAND archives and report every available matched pair."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile


def main():
    root = Path(__file__).resolve().parents[2] / 'reports/hfh_colab_exploration_20261111'
    out = root / 'bands_analysis'
    rows = []
    for path in sorted((out / 'current').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for name, digest in json.loads(archive.read('results/archive_manifest.json')).items():
                assert hashlib.sha256(archive.read('results/' + name)).hexdigest() == digest
            row = json.loads(archive.read('results/paired_comparison.json'))
            calibration = json.loads(archive.read('results/calibration.json'))
            curve = list(csv.DictReader(io.StringIO(archive.read('results/metrics.csv').decode())))
            row['calibration_ratio'] = calibration['statistics']['bands']['weighted_median_ratio']
            control = root / 'recipe_analysis' / path.name
            if not control.exists():
                control = root / 'recipe_analysis' / ('swinunetr_lr0.001_poly_cap__' + path.name)
            assert hashlib.sha256(control.read_bytes()).hexdigest() == row['baseline_archive_sha256']
            with zipfile.ZipFile(control) as baseline:
                name = next(n for n in baseline.namelist() if n.endswith('/metrics.csv'))
                bc = list(csv.DictReader(io.StringIO(baseline.read(name).decode())))
            by_step = {int(x['step']): float(x['macro_hard_dice']) for x in bc}
            differences = [abs(by_step[int(x['step'])] - float(x['macro_hard_dice']))
                           for x in curve if int(x['step']) <= calibration['anchor_successful_updates']
                           and int(x['step']) in by_step]
            assert differences and max(differences) < 1e-12
            assert abs(row['delta_dice_pp'] - 100*(row['band_dice']-row['baseline_dice'])) < 1e-9
            assert row['optimization_flops'] <= 31976450667090
            rows.append(row)
    models = ['nnunet', 'unetr_vit_b16', 'basic_unet16', 'swinunetr']
    summary = []
    for model in models:
        group = [r for r in rows if r['model'] == model]
        if not group:
            continue
        mean = lambda key: statistics.mean(r[key] for r in group)
        item = dict(model=model, completed=len(group), baseline=100*mean('baseline_dice'),
                    band=100*mean('band_dice'), delta_pp=mean('delta_dice_pp'),
                    improved=sum(r['improved'] for r in group),
                    train_delta_pp=100*(mean('band_training_dice')-mean('baseline_training_dice')))
        for error in ('fp', 'fn'):
            control = sum(r['baseline_foreground_'+error] for r in group)
            band = sum(r['band_foreground_'+error] for r in group)
            item[error+'_change_pct'] = 100*(band/control-1)
        summary.append(item)
    (out/'verified_pairs.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out/'paired_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = np.full((9, 4), np.nan)
    for row in rows:
        values[row['fold']*3+row['seed'], models.index(row['model'])] = row['delta_dice_pp']
    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(values, cmap=cmap, vmin=-11, vmax=11, aspect='auto')
    for i in range(9):
        for j in range(4):
            value = values[i, j]
            ax.text(j, i, 'pending' if np.isnan(value) else f'{value:+.2f}',
                    ha='center', va='center', color='white' if abs(value)>6 else '#202020')
    ax.set_xticks(range(4), ['nnU-Net', 'UNETR', 'BasicUNet', 'SwinUNETR'])
    ax.set_yticks(range(9), [f'Fold {f} · seed {s}' for f in range(3) for s in range(3)])
    ax.set_title(f'HFH BAND − matched baseline: validation Dice points\n{len(rows)}/36 completed and archive-verified', pad=15)
    fig.colorbar(im, ax=ax, label='Dice percentage points')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'band_delta_matrix.{extension}', dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
