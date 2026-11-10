"""Plot matched MSD 3D UNet constraint curves and final Dice changes."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.cross_dataset_campaign.common import read


def curve(path):
    config=read(path/'config.json')
    with (path/'metrics.csv').open() as stream:rows=list(csv.DictReader(stream))
    return ([100*float(r['estimated_training_flops'])/config['budget_flops'] for r in rows],
            [100*float(r['macro_hard_dice']) for r in rows])


def main():
    root=Path(__file__).resolve().parents[2]/'reports/msd_unet_ce_constraints_20261014'
    report=read(root/'comparison.json')
    if report['status']!='complete':raise ValueError('All six verified results required')
    baseline=root.parent/'msd_other_models_loss_pairs_20261014/runs/MSD_basic_unet16_dice_ce_f0s0'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.6,5.5),sharex=True,sharey=True)
    for ax,(arm,title) in zip(axes,[('dice_bands','Bands'),('dice_bands_edge_depth','Bands + original edge + depth CE')]):
        ax.plot(*curve(baseline),color='#333333',label='Dice+CE baseline',linewidth=2,linestyle='--')
        for target,color in [(0.1,'#2864a0'),(0.4,'#d27921'),(0.6,'#7d4d91')]:
            row=next(r for r in report['runs'] if r['objective']==arm and r['target']==target)
            x,y=curve(root/'runs'/row['id'])
            ax.plot(x,y,color=color,linewidth=2,label=f'{int(target*100)}% target: {y[-1]:.2f}% Dice')
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlim(0,101);ax.set_ylim(0,100)
        ax.grid(axis='y',alpha=.2);ax.set_xlabel('Training FLOP budget consumed (%)')
        ax.legend(frameon=False,loc='lower right',fontsize=9)
    axes[0].set_ylabel('Validation Dice (%)')
    fig.suptitle('3D UNet: constraints on top of Dice+CE',x=.07,ha='left',fontsize=16,fontweight='bold')
    fig.text(.07,.91,'MSD · fold 0 · seed 0 · 10 training / 52 validation volumes · no augmentation')
    fig.text(.07,.025,'All final-budget checkpoints. Targets use sequential training-only gradient calibration; p95 ceilings may limit achieved ratios.',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,bottom=.14,top=.81,wspace=.13)
    for suffix in ('png','pdf'):fig.savefig(root/f'constraint_comparison.{suffix}',dpi=180)
    print(root/'constraint_comparison.png')


if __name__=='__main__':main()
