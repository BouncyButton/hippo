"""Verify and summarize the selected HFH recipes without dropping regressions."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import zipfile


def main():
    root=Path(__file__).resolve().parents[2]/'reports/hfh_colab_exploration_20261111'
    analysis=root/'bands_analysis'
    locations={
        'nnunet':('nnunet_r12',analysis/'selected_recipes/nnunet_r12'),
        'unetr_vit_b16':('bands_v1',analysis/'current'),
        'basic_unet16':('basic_unet16_r09',analysis/'selected_recipes/basic_unet16_r09'),
        'swinunetr':('bands_v1',analysis/'current'),
    }
    rows=[];summaries=[]
    for model,(recipe,folder) in locations.items():
        group=[]
        for path in sorted(folder.glob(model+'_f*_s*.zip')):
            with zipfile.ZipFile(path) as archive:
                for name,digest in json.loads(archive.read('results/archive_manifest.json')).items():
                    assert hashlib.sha256(archive.read('results/'+name)).hexdigest()==digest
                row=json.loads(archive.read('results/paired_comparison.json'))
                protocol=json.loads(archive.read('protocol/protocol.json'))
                config=json.loads(archive.read('results/config.json'))
                control=root/'recipe_analysis'/path.name
                if not control.exists():control=root/'recipe_analysis'/('swinunetr_lr0.001_poly_cap__'+path.name)
                assert hashlib.sha256(control.read_bytes()).hexdigest()==row['baseline_archive_sha256']
                with zipfile.ZipFile(control) as baseline:
                    assert archive.read('results/initialization.json')==baseline.read('results/initialization.json')
                    baseline_config=json.loads(baseline.read('results/config.json'))
                    for key in ('model_identity','seed','fold','dataset_sha256','train_cases','validation_cases','lr_horizon_updates','initial_lr','augmentation','precision','max_steps'):
                        assert config[key]==baseline_config[key]
                assert abs(row['delta_dice_pp']-100*(row['band_dice']-row['baseline_dice']))<1e-9
                assert row['optimization_flops']<=31976450667090
                row['recipe']=recipe;group.append(row)
        assert len(group)==9, (model,len(group))
        assert {(r['fold'],r['seed']) for r in group}=={(f,s) for f in range(3) for s in range(3)}
        fold_deltas={f:statistics.mean(r['delta_dice_pp'] for r in group if r['fold']==f) for f in range(3)}
        assert min(fold_deltas.values())>0
        assert min(r['band_dice'] for r in group)>=.5
        schedule=protocol['models'][protocol['trainer_alias']]['objectives']['dice_bands']
        summary=dict(model=model,recipe=recipe,baseline_dice=100*statistics.mean(r['baseline_dice'] for r in group),
            selected_dice=100*statistics.mean(r['band_dice'] for r in group),delta_pp=statistics.mean(r['delta_dice_pp'] for r in group),
            improved_pairs=sum(r['improved'] for r in group),minimum_dice=100*min(r['band_dice'] for r in group),fold_deltas=fold_deltas,
            initial_lr=config['initial_lr'],updates=config['max_steps'],native_prefix=schedule['prefix_updates'],ramp=schedule['constraint_warmup_updates'],
            calibration_target=protocol['calibration_target_ratio'],calibration_p95_cap=protocol['calibration_p95_cap'],
            training_delta_pp=100*statistics.mean(r['band_training_dice']-r['baseline_training_dice'] for r in group),
            optimization_flops=schedule['consumed_training_flops'],calibration_flops=schedule['calibration_flops'])
        for error in ('fp','fn'):
            b=sum(r['baseline_foreground_'+error] for r in group);a=sum(r['band_foreground_'+error] for r in group)
            summary[error+'_change_pct']=100*(a/b-1)
        summaries.append(summary);rows.extend(group)
    target=analysis/'selected_recipes';target.mkdir(exist_ok=True)
    result=dict(selection='Exploratory validation-selected recipes; every fold mean positive, all nine Dice >=.5; individual regressions retained',models=summaries)
    (target/'winning_recipes.json').write_text(json.dumps(result,indent=2)+'\n')
    with (target/'all_36_selected_pairs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=list(locations);values=np.full((9,4),np.nan)
    for row in rows:values[row['fold']*3+row['seed'],models.index(row['model'])]=row['delta_dice_pp']
    fig,ax=plt.subplots(figsize=(10,6.8));im=ax.imshow(values,cmap='RdBu',vmin=-11,vmax=11,aspect='auto')
    for i in range(9):
        for j in range(4):ax.text(j,i,f'{values[i,j]:+.3f}',ha='center',va='center',color='white' if abs(values[i,j])>6 else '#202020')
    ax.set_xticks(range(4),['nnU-Net','UNETR','BasicUNet','SwinUNETR']);ax.set_yticks(range(9),[f'Fold {f} · seed {s}' for f in range(3) for s in range(3)])
    ax.set_title('Selected HFH BAND recipes − frozen baselines\nAll 36 pairs verified; validation-selected, regressions retained',pad=14)
    fig.colorbar(im,ax=ax,label='Dice percentage points');fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(target/f'selected_recipe_deltas.{extension}',dpi=160)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
