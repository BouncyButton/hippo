"""Standalone research figures for the read-only audit."""
import os
os.environ['MPLCONFIGDIR']='/private/tmp/geofwi-audit-mpl'
os.environ['XDG_CACHE_HOME']='/private/tmp/geofwi-audit-cache'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'results_audit.json').read_text())
rows=sorted([r for r in report['runs'] if r['model']=='unetr_vit_b16'],key=lambda r:r['seed'])
fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),layout='constrained')
ax=axes[0]
for folder in sorted((HERE/'evidence').glob('GeoFWI3D_unetr*')):
    curve=list(csv.DictReader((folder/'metrics.csv').open()))
    ax.plot([float(r['step']) for r in curve],[100*float(r['macro_hard_dice']) for r in curve],
            color='#5681a8',alpha=.25,lw=.8)
mean=report['mean_curve'];ax.plot([r['step'] for r in mean],[100*r['dice_mean'] for r in mean],
    color='#123e68',lw=2.5,label='Mean of 25 seeds')
ax.axhline(50,color='#b43432',ls='--',lw=1.5,label='Required >50%')
ax.set(xlabel='Optimizer updates (414 = full cap)',ylabel='Validation Dice (%)',ylim=(0,60),
       title='Every seed peaks below the target')
ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.16)
ax=axes[1];seeds=[r['seed'] for r in rows]
ax.plot(seeds,[100*r['train_dice'] for r in rows],'o-',color='#247559',lw=1,label='Final training')
ax.plot(seeds,[100*r['best_dice'] for r in rows],'s--',color='#ce8b28',lw=1,label='Best validation')
ax.plot(seeds,[100*r['final_dice'] for r in rows],'o-',color='#123e68',lw=1,label='Final validation')
ax.axhline(50,color='#b43432',ls='--',lw=1.5)
ax.set(xlabel='Seed',ylabel='Mean per-volume Dice (%)',ylim=(0,90),title='Large training–validation gap',xticks=range(0,25,4))
ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.16)
fig.suptitle('GeoFWI3D · UNETR · fold 0 · 19 training / 96 validation volumes\nNo augmentation · 182,178.13 GFLOP cap · seeds 0–24',fontsize=13)
fig.savefig(HERE/'unetr_audit_curves.png',dpi=180)
fig.savefig(HERE/'unetr_audit_curves.pdf')
plt.close(fig)
print('Saved PNG and PDF figures',flush=True)
