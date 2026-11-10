"""Compare training and validation curves at matched optimization FLOPs."""
import csv
from pathlib import Path
import numpy as np

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive

ROOT=Path(__file__).resolve().parents[2]
NAME='mni_shared_cap3_bands_f0s0_20261107'
LABELS={'nnunet':'nnUNet','swinunetr':'SwinUNETR','unetr_vit_b16':'UNETR','basic_unet16':'UNet'}


def curve(folder):
    verify_archive(folder,read(folder/'archive_manifest.json'))
    with (folder/'metrics.csv').open() as stream:
        unique={int(r['charged_attempts']):r for r in csv.DictReader(stream)}
    rows=[unique[k] for k in sorted(unique)]
    values=np.array([[float(r[k]) for k in ('compute_fraction','training_macro_hard_dice','macro_hard_dice')] for r in rows])
    assert (np.diff(values[:,0])>0).all()
    return values*np.array([1,100,100])


def main():
    report=ROOT/'reports'/NAME
    controls={r['model']:Path(r['archive']) for r in read(report/'frozen/baseline_reuse.json')}
    grid=np.unique(np.r_[np.arange(.15,.96,.05),2/3,5/6])
    results=[];curves={}
    for model,folder in controls.items():
        baseline=curve(folder);bands=curve(report/'runs'/f'MNI_{model}_dice_bands_f0s0')
        assert max(baseline[0,0],bands[0,0])<=min(grid) and max(grid)<=min(baseline[-1,0],bands[-1,0])
        a=np.column_stack([np.interp(grid,baseline[:,0],baseline[:,i]) for i in (1,2)])
        b=np.column_stack([np.interp(grid,bands[:,0],bands[:,i]) for i in (1,2)])
        delta=b[:,1]-a[:,1];peak=int(delta.argmax());arms={}
        for name,z in [('baseline',baseline),('bands',bands)]:
            k=int(z[:,2].argmax())
            arms[name]=dict(final_training_dice=z[-1,1],final_validation_dice=z[-1,2],
                best_recorded_fraction=z[k,0],best_recorded_validation_dice=z[k,2],training_at_best_validation=z[k,1],
                validation_peak_to_final_drop_pp=z[k,2]-z[-1,2],
                last20_training_gain_pp=z[-1,1]-np.interp(.8,z[:,0],z[:,1]),
                last20_validation_gain_pp=z[-1,2]-np.interp(.8,z[:,0],z[:,2]))
        results.append(dict(model=model,fractions=grid.tolist(),baseline_training=a[:,0].tolist(),bands_training=b[:,0].tolist(),
                            baseline_validation=a[:,1].tolist(),bands_validation=b[:,1].tolist(),delta_pp=delta.tolist(),
                            peak_sampled_gain_pp=delta[peak],peak_sampled_gain_fraction=grid[peak],
                            final_gain_pp=bands[-1,2]-baseline[-1,2],arms=arms,
                            baseline_metrics_sha256=digest(folder/'metrics.csv'),
                            bands_metrics_sha256=digest(report/'runs'/f'MNI_{model}_dice_bands_f0s0'/'metrics.csv')))
        curves[model]=(baseline,bands)
    write(report/'learning_curve_comparison.json',dict(results=results,
          matching='Linear interpolation between saved evaluations at common fractions of the3x optimization cap. Final metrics are actual final-budget evaluations.',
          caveats=['Single fold0seed0 exploratory curves; peaks are post-hoc descriptions.',
                   'Earlier checkpoints retain the3x LR horizon. Fresh shorter-cap runs with matched schedules can behave differently.']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for ax,r in zip(axes.flat,results):
        ax.axhline(0,color='#707070',lw=1)
        ax.axvspan(10,20,color='#dfe7ef',alpha=.7)
        ax.plot(np.r_[grid*100,100],r['delta_pp']+[r['final_gain_pp']],color='#147d92',marker='o',markersize=3)
        ax.set_title(LABELS[r['model']],loc='left',weight='bold')
        ax.set(xlim=(10,101),ylabel='Validation Dice gain (pp)',xlabel='Current 3× FLOP cap used (%)')
        ax.grid(alpha=.18)
    fig.suptitle('MNI: bands minus baseline — fold 0, seed 0\nShaded interval: constraint ramp',fontsize=14)
    fig.tight_layout();fig.savefig(report/'bands_gain_curves.png',dpi=170);fig.savefig(report/'bands_gain_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,(model,(baseline,bands)) in zip(axes.flat,curves.items()):
        for label,z,color in [('Baseline',baseline,'#2459a6'),('Bands',bands,'#d57625')]:
            ax.plot(100*z[:,0],z[:,2],color=color,label=label+' validation')
            ax.plot(100*z[:,0],z[:,1],color=color,ls='--',alpha=.8,label=label+' training')
        ax.axhline(50,color='#777777',lw=.8,ls=':');ax.set_title(LABELS[model],loc='left',weight='bold')
        ax.set(ylim=(0,100),xlabel='Current 3× FLOP cap used (%)',ylabel='Macro Dice (%)');ax.grid(alpha=.18)
        ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('MNI: training fit and validation performance',fontsize=15)
    fig.tight_layout();fig.savefig(report/'training_validation_curves.png',dpi=170);fig.savefig(report/'training_validation_curves.pdf');plt.close(fig)
    print('Saved verified curve analysis and two figures.')


if __name__=='__main__':main()
