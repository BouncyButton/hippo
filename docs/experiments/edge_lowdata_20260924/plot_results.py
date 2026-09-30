"""Plot saved historical and new learning curves; no model inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
results = json.loads((root/'RESULTS.json').read_text())
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex='col', gridspec_kw={'height_ratios':[2,1]})
for col, seed in enumerate(('0','1','2')):
    record = results['seeds'][seed]
    histories = record['histories.json']
    for model, label, color in [('bands','Bands only','#6c7086'), ('edge','Bands + edge','#007f86')]:
        rows = histories[model]
        axes[0,col].plot([int(r['epoch']) for r in rows], [100*float(r['val_dice_hard']) for r in rows], label=label, color=color, lw=1.8)
    target = record['learning_curves.json']['thresholds']['bands_reference_best']
    axes[0,col].axhline(target*100, color='#6c7086', ls=':', lw=1)
    axes[0,col].set_title(f'Seed {seed}')
    axes[0,col].set_ylim(0,85)
    n = min(len(histories['bands']),len(histories['edge']))
    delta = [100*(float(histories['edge'][i]['val_dice_hard'])-float(histories['bands'][i]['val_dice_hard'])) for i in range(n)]
    axes[1,col].plot(range(1,n+1),delta,color='#007f86',lw=1.5)
    axes[1,col].axhline(0,color='#6c7086',lw=.8)
    axes[1,col].set_xlabel('Epoch (10 updates per epoch)')
    for row in (0,1):
        axes[row,col].grid(alpha=.18)
        axes[row,col].spines[['top','right']].set_visible(False)
        axes[row,col].set_xlim(1,75)
axes[0,0].set_ylabel('Validation macro Dice (%)')
axes[1,0].set_ylabel('Edge − bands (pp)')
axes[0,0].legend(frameon=False,loc='lower right')
fig.suptitle('5% training data · fold 0 · bands-only versus bands + edge',fontsize=14)
fig.tight_layout()
fig.savefig(root/'learning_curves.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'learning_curves.pdf',bbox_inches='tight')
