"""Plot archived first-pilot learning curves without touching live analysis."""
import csv
from scripts.cross_dataset_campaign.common import read
from scripts.mni_constraints import unetr_simple_recipe_study as study


def plot_first_pilot(report, archived, references):
    """Show training fit and validation separately on the first fixed pilot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seed,fold = study.PILOTS[0]
    fig,axes = plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    labels={'dice':'Fresh baseline','dice_bands':'Delayed bands','dice_bands_edge_sdf_depth':'Delayed full combination'}
    for column,recipe in enumerate(study.RECIPES):
        curves=[('Current winning baseline',references[seed,fold],'black','--')]
        for arm,color in zip(study.ARMS,('C0','C1','C2')):
            identifier=study.identifier(recipe['key'],arm,seed,fold)
            if any(r['id']==identifier for r in archived):
                curves.append((labels[arm],report/'runs'/identifier,color,'-'))
        for label,folder,color,style in curves:
            with (folder/'metrics.csv').open() as stream:
                curve=list(csv.DictReader(stream))
            for row,field in enumerate(('training_macro_hard_dice','macro_hard_dice')):
                axes[row,column].plot([int(r['step']) for r in curve],
                    [100*float(r[field]) for r in curve],label=label,color=color,linestyle=style)
        axes[0,column].set_title(f"{recipe['key']}: dropout {recipe['dropout_rate']}, augmentation {recipe['augmentation']}")
        for row in range(2):
            axes[row,column].axvline(102,color='gray',linestyle=':',alpha=.6)
            axes[row,column].set_ylim(0,100)
            axes[row,column].grid(alpha=.2)
        axes[1,column].set_xlabel('Optimizer updates')
        axes[1,column].legend(fontsize=8)
    axes[0,0].set_ylabel('Training macro Dice (%)')
    axes[1,0].set_ylabel('Validation macro Dice (%)')
    fig.suptitle(f'UNETR base-recipe interaction: seed {seed}, fold {fold} (first development pilot)\nVerified completed runs only; vertical line marks constraint calibration/onset')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(report/f'first_pilot_learning_curves.{suffix}',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    report=study.ROOT/'reports'/study.NAME
    frozen=report/'frozen'
    references={(c['seed'],c['fold']):frozen/'controls'/c['name'] for c in read(frozen/'controls.json')}
    plot_first_pilot(report,read(report/'simple_recipe_results.json')['archived_runs'],references)
