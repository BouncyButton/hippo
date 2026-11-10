"""Reproduce the curve/gradient audit figure with explicit probe limitations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[2]/'reports/mni_n10_constraints14_shared43t_3fold3seed_20261107'
    data=json.loads((root/'unetr_gradient_strength_audit_20261108.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,.10,1,.90))
    for arm,label,color in [('dice','Baseline','#24758e'),('dice_bands','Bands, 5% target','#c56039')]:
        row=next(x for x in data['curve_audit']['summary'] if x['arm']==arm)
        x=[25,50,75,100]
        for split,style in [('val','-'),('train','--')]:
            values=[row['milestones'][str(q/100)][split] for q in x]
            axes[0].plot(x,values,style,marker='o',color=color,label=f'{label}, {split}')
        axes[1].plot(x,[row['milestones'][str(q/100)]['val'] for q in x],'-o',color=color,label=label)
    axes[0].set(title='Train and validation Dice',ylabel='Dice (%)',xlabel='Optimization budget (%)')
    axes[0].legend(fontsize=7)
    axes[1].set(title='Validation near the end',ylabel='Dice (%)',xlabel='Optimization budget (%)',xlim=(70,102),ylim=(62,69))
    axes[1].legend(fontsize=8)
    axes[2].plot([r['quarter'] for r in data['summary']],
                 [100*r['mean_training_median_weighted_logit_ratio'] for r in data['summary']],'-o',color='#c56039')
    axes[2].axhline(5,color='#666666',linestyle='--',label='Calibration target')
    axes[2].set(title='Training logit-gradient ratio',ylabel='Weighted bands / base norm (%)',xlabel='Optimization budget (%)',ylim=(0,6))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    fig.suptitle('UNETR: nine fixed fold/seed pairs at the shared 43T cap',fontsize=13)
    fig.text(.5,.025,'Dice: means across nine runs. Gradient: mean of per-run training-crop medians; unaugmented evaluation-mode logit probes, not optimizer-update ratios.',ha='center',fontsize=8)
    for extension in ('png','pdf'):
        fig.savefig(root/f'unetr_overfitting_gradient_audit_20261108.{extension}',dpi=170)


if __name__=='__main__':main()
