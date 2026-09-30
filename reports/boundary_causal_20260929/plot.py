"""Research figures from case-level identity audits and saved learning curves."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
colors={'sum':'#2463a6','dice':'#64748b','sum_aug':'#e07b24','dice_aug':'#14856e','pcgrad':'#a44b77'}
fig,ax=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
for arm,color in colors.items():
    folder=(ROOT/'results/seed0'/arm) if arm!='pcgrad' else ROOT.parent/'pcgrad_50cases_seed0_20260929/results/fold0/seed0/pcgrad'
    if not (folder/'epochs').exists():continue
    rows=[read(p) for p in sorted((folder/'epochs').glob('*.json'))]
    ax[0].plot([r['epoch']*50 for r in rows],[r['val_dice_hard']*100 for r in rows],label=arm,color=color,lw=1.7)
    audits=folder/('audits' if arm!='pcgrad' else 'coherence')
    points=[]
    for p in sorted(audits.glob('epoch_*.json')):
        epoch=int(p.stem.split('_')[-1]); rr=read(p)
        for split in ('train','validation'):
            m=[r['metrics'] for r in rr if r['split']==split]
            if m:points.append((epoch*50,split,np.mean([x['macro_dice'] for x in m])*100,
                np.mean([x['inner_fn']+x['outer_fp'] for x in m])))
    for split,ls in [('train','--'),('validation','-')]:
        p=[r for r in points if r[1]==split]
        if p:
            ax[1].plot([r[0] for r in p],[r[2] for r in p],color=color,ls=ls,lw=1.7)
            ax[2].plot([r[0] for r in p],[r[3] for r in p],color=color,ls=ls,lw=1.7)
ax[0].set(title='Validation learning curves',ylabel='Macro A/P Dice (%)',ylim=(20,100))
ax[0].legend(frameon=False,fontsize=9)
ax[1].set(title='Identity-input fit and generalization',ylabel='Macro A/P Dice (%)')
ax[2].set(title='Unique boundary errors per case',ylabel='Inner FN + outer FP')
for a in ax:
    a.set_xlabel('Optimizer updates attempted')
    a.grid(axis='y',alpha=.18)
    a.spines[['top','right']].set_visible(False)
fig.suptitle('Matched 50-case experiment · solid validation, dashed training · same 52 validation cases',fontsize=13)
fig.savefig(ROOT/'trajectories.png',dpi=180)
fig.savefig(ROOT/'trajectories.pdf')
plt.close(fig)
