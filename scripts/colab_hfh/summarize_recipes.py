"""Verify downloaded archives and summarize all nine fixed cells per recipe."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    seen = set()
    for path in sorted(args.directory.glob('*.zip')):
        with zipfile.ZipFile(path) as z:
            if 'results/archive_manifest.json' not in z.namelist():
                continue
            manifest = json.loads(z.read('results/archive_manifest.json'))
            for filename, expected in manifest.items():
                if hashlib.sha256(z.read('results/'+filename)).hexdigest() != expected:
                    raise ValueError('Corrupt result: '+str(path))
            result = json.loads(z.read('results/result.json'))
            protocol = json.loads(z.read('protocol/protocol.json'))
            binding = json.loads(z.read('results/runtime_recipe.json'))
            if hashlib.sha256(z.read('protocol/protocol.json')).hexdigest() != binding['protocol_sha256']:
                raise ValueError('Protocol binding mismatch: '+str(path))
            rate = protocol['initial_lrs'][protocol['trainer_alias']]
            key = (result['model'], rate, result['fold'], result['seed'])
            if key in seen:
                continue
            seen.add(key)
            training = json.loads(z.read('results/training_final.json'))
            validation = json.loads(z.read('results/validation_latest.json'))
            timing = json.loads(z.read('results/worker_timing.json'))
            rows.append(dict(model=key[0], initial_lr=rate, fold=key[2], seed=key[3],
                dice=result['validation_dice'], train_dice=training['summary']['macro_hard_dice'],
                soft_dice=validation['summary']['macro_soft_dice'],
                false_positive_voxels=sum(c['foreground_fp'] for c in validation['cases']),
                false_negative_voxels=sum(c['foreground_fn'] for c in validation['cases']),
                updates=result['updates'], optimization_flops=result['optimization_flops'],
                worker_seconds=timing['seconds'], peak_allocated_bytes=timing['peak_allocated_bytes'],
                archive=path.name, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summaries = []
    for model, rate in sorted({(r['model'],r['initial_lr']) for r in rows}):
        group = [r for r in rows if (r['model'],r['initial_lr']) == (model,rate)]
        full = {(r['fold'],r['seed']) for r in group} == {(f,s) for f in range(3) for s in range(3)}
        summaries.append(dict(model=model, initial_lr=rate, cells=len(group),
            passed_all_nine=full and all(r['dice'] >= .5 for r in group),
            minimum_dice=min(r['dice'] for r in group), mean_dice=statistics.mean(r['dice'] for r in group),
            std_across_nine=statistics.stdev(r['dice'] for r in group) if len(group)>1 else None,
            mean_training_dice=statistics.mean(r['train_dice'] for r in group),
            mean_worker_seconds=statistics.mean(r['worker_seconds'] for r in group),
            mean_by_seed={s:statistics.mean(r['dice'] for r in group if r['seed']==s)
                          for s in range(3) if any(r['seed']==s for r in group)}))
    report = dict(selection='Exploratory validation-selected recipes; not an unbiased test estimate',
        threshold=.5, seeds=[0,1,2], folds=[0,1,2], summaries=summaries, runs=rows)
    (args.directory/'verified_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        with (args.directory/'verified_runs.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    qualified = [s for s in summaries if s['passed_all_nine']]
    if qualified:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        names = {'nnunet':'nnU-Net', 'unetr_vit_b16':'UNETR',
                 'basic_unet16':'BasicUNet', 'swinunetr':'SwinUNETR'}
        matrix = np.array([[100*next(r['dice'] for r in rows if r['model']==s['model']
            and r['initial_lr']==s['initial_lr'] and r['fold']==f and r['seed']==seed)
            for f in range(3) for seed in range(3)] for s in qualified])
        fig, ax = plt.subplots(figsize=(11, max(2.6,len(qualified)*.8+1.5)))
        image = ax.imshow(matrix, cmap='YlGn', vmin=50, vmax=90, aspect='auto')
        ax.set_xticks(range(9), [f'F{f} / S{s}' for f in range(3) for s in range(3)])
        ax.set_yticks(range(len(qualified)), [names[s['model']] for s in qualified])
        for row in range(len(qualified)):
            for col in range(9):
                ax.text(col,row,f'{matrix[row,col]:.1f}',ha='center',va='center',
                        color='white' if matrix[row,col]>77 else '#172019',fontsize=11)
        ax.set_title('HFH · final validation Dice (%) · same seeds and folds',pad=14)
        fig.colorbar(image, ax=ax, label='Dice (%)')
        fig.text(.5,.02,'Each cell must be ≥50%. Recipes selected on validation; no unbiased test claim.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,1))
        fig.savefig(args.directory/'qualified_recipes.png',dpi=180)
        fig.savefig(args.directory/'qualified_recipes.pdf')
        plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
