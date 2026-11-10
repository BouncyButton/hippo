import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

root=Path(__file__).resolve().parent
data=json.loads((root/'analysis.json').read_text())
rows=data['aggregate']
models=['nnunet','swinunetr','unetr_vit_b16','basic_unet16']
names=['nnUNet','SwinUNETR','UNETR','3D U-Net']
arms=[r['arm'] for r in rows if r['model']==models[0]]
labels=[r['condition'] for r in rows if r['model']==models[0]]
get=lambda m,a:next(r for r in rows if r['model']==m and r['arm']==a)
values=np.array([[get(m,a)['gain_pp'] for m in models] for a in arms])
fig,ax=plt.subplots(figsize=(12,9),facecolor='white')
im=ax.imshow(values,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=6),aspect='auto')
ax.set_xticks(range(4),names,fontsize=12)
ax.set_yticks(range(len(arms)),labels,fontsize=11)
ax.xaxis.tick_top();ax.tick_params(length=0,pad=11)
for i,a in enumerate(arms):
    for j,m in enumerate(models):
        r=get(m,a)
        text=f"{r['validation_dice']:.2f}%"+('  baseline' if i==0 else f"\n{r['gain_pp']:+.2f} pp")
        ax.text(j,i,text,ha='center',va='center',fontsize=10.5,color='#172b36',fontweight='bold' if data['best'][m]['arm']==a else 'normal')
for x in np.arange(-.5,4,1):ax.axvline(x,color='white',linewidth=3)
for y in np.arange(-.5,len(arms),1):ax.axhline(y,color='white',linewidth=3)
for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('ADNI • 504 completed runs • no augmentation',fontsize=19,x=.07,ha='left',y=.98,fontweight='bold')
fig.text(.07,.925,'Validation Dice and gain against the matched baseline. Each cell averages 3 folds × 3 seeds.',fontsize=11,color='#526372')
fig.text(.07,.035,'Folds 0/1/2; seeds 0/1/4. Seed replacement was post hoc: exploratory results.\nnnUNet: Dice+CE, 60% targets. Other models: Dice, 10% targets. Calibration FLOPs are separate.',fontsize=10,color='#526372')
fig.subplots_adjust(left=.29,right=.98,top=.86,bottom=.11)
fig.savefig(root/'ADNI_dice_and_gains.png',dpi=160)
fig.savefig(root/'ADNI_dice_and_gains.pdf')
