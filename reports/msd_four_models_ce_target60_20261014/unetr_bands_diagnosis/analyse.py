import csv,json,sys
from pathlib import Path
root=Path('/Users/filippofocaccia/Desktop/hippo/reports/msd_four_models_ce_target60_20261014')
sys.path.insert(0,str(root/'frozen/source'))
from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.cross_dataset_campaign.storage import verify_archive
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss,build_boundary_bands
out=root/'unetr_bands_diagnosis'
records={};curves={};configs={};initializations={}
for arm in ['dice','dice_bands']:
 p=root/'runs'/f'MSD_unetr_vit_b16_ce_{arm}_target60_f0s0'
 verify_archive(p,read(p/'archive_manifest.json'))
 v=read(p/'validation_latest.json');d=read(p/'completion.json')
 c=read(p/'config.json');configs[arm]=c;initializations[arm]=read(p/'initialization.json')
 curves[arm]=list(csv.DictReader((p/'metrics.csv').open()))
 cm=[[sum(x['confusion'][i][j] for x in v['cases']) for j in range(3)] for i in range(3)]
 union=[]
 for x in v['cases']:
  m=x['confusion'];tp=sum(m[i][j] for i in [1,2] for j in [1,2]);fp=m[0][1]+m[0][2];fn=m[1][0]+m[2][0]
  union.append(2*tp/(2*tp+fp+fn))
 records[arm]=dict(validation_macro_dice_percent=100*v['summary']['macro_hard_dice'],
  validation_anterior_dice_percent=100*v['summary']['anterior_hard_dice'],validation_posterior_dice_percent=100*v['summary']['posterior_hard_dice'],
  training_macro_dice_percent=100*d['final_training_hard_dice'],best_macro_dice_percent=100*d['best_validation_hard_dice'],
  binary_foreground_dice_percent=100*sum(union)/len(union),foreground_fn=cm[1][0]+cm[2][0],foreground_fp=cm[0][1]+cm[0][2],
  anterior_to_posterior=cm[1][2],posterior_to_anterior=cm[2][1],confusion=cm,updates=d['updates'],discarded_attempts=d['discarded_attempts'])
assert initializations['dice']==initializations['dice_bands']
config_differences={k:[configs['dice'].get(k),configs['dice_bands'].get(k)] for k in configs['dice'] if configs['dice'].get(k)!=configs['dice_bands'].get(k)}
assert set(config_differences)=={'objective','schedule','sdf_prefix_updates'}
assert all(r['discarded_attempts']==0 and r['updates']==245 for r in records.values())
prefix=[]
for a,b in zip(curves['dice'],curves['dice_bands']):
 if int(a['step'])>50:break
 for key in ['step','mean_train_loss','learning_rate','macro_hard_dice','anterior_hard_dice','posterior_hard_dice']:
  assert a[key]==b[key],key
 prefix.append(int(a['step']))
# Reproduce the mathematical failure mode using the exact archived bands implementation.
torch.set_num_threads(2);torch.manual_seed(0)
labels=torch.zeros(1,1,16,16,16,dtype=torch.long);labels[:,:,4:12,4:12,4:12]=1;labels[:,:,8:12,4:12,4:12]=2
z=torch.randn(1,3,16,16,16,requires_grad=True)
loss=OuterBoundaryBandLoss()(z,labels).loss
swapped=OuterBoundaryBandLoss()(z[:,[0,2,1]],labels).loss
flipped_labels=torch.where(labels==1,2,torch.where(labels==2,1,0))
label_swapped=OuterBoundaryBandLoss()(z,flipped_labels).loss
assert torch.equal(loss,swapped) and torch.equal(loss,label_swapped)
z=torch.zeros(1,3,16,16,16);z[:,2]=2.;z.requires_grad_(True)
loss=OuterBoundaryBandLoss()(z,labels).loss;g=torch.autograd.grad(loss,z)[0]
inner,_=build_boundary_bands(labels>0);i=tuple(((inner[:,0])&(labels[:,0]==1)).nonzero()[0].tolist())
b,x,y,zindex=i;ga=float(g[b,1,x,y,zindex]);gp=float(g[b,2,x,y,zindex])
assert ga<0 and gp<ga
mechanism=dict(logit_AP_swap_invariant=True,label_AP_swap_invariant=True,
 synthetic_wrong_posterior_to_correct_anterior_gradient_ratio=abs(gp/ga),
 interpretation='On an inner-band anterior voxel with posterior logit already larger, bands raises both foreground logits but pushes the already larger posterior more strongly. It cannot penalize A/P swaps. This verifies a possible mechanism, not its causal contribution throughout training.')
cal=read(root/'runs/MSD_unetr_vit_b16_ce_dice_bands_target60_f0s0/calibration.json')
result=dict(status='evidence_complete_causal_attribution_provisional',records=records,initialization_match=True,
 config_differences=config_differences,prefix_metrics_exactly_match_at_updates=prefix,
 calibration=cal,synthetic_test=mechanism,
 limits=['One fold/seed; do not infer an inherent transformer weakness.',
 'Calibration measures detached logit-gradient RMS at update50, then holds a scalar weight fixed. Later parameter-gradient ratios were not recorded.',
 'Completed checkpoints were removed after verified archive under storage policy; no retrospective model-gradient test was performed.'],
 source_sha256=digest(root/'frozen/source/thesis/new_constraints/bands/outer_boundary.py'))
write(out/'diagnosis.json',result)
fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharex=True,sharey=True)
for ax,key,title in zip(axs,['macro_hard_dice','anterior_hard_dice','posterior_hard_dice'],['Mean A/P Dice','Anterior Dice','Posterior Dice']):
 for arm,label,color in [('dice','Dice + CE','#2563eb'),('dice_bands','Dice + CE + bands 60%','#dc2626')]:
  ax.plot([int(r['step']) for r in curves[arm]],[100*float(r[key]) for r in curves[arm]],label=label,color=color,linewidth=2)
 ax.axvspan(50,100,color='gray',alpha=.12);ax.set_title(title);ax.set_xlabel('Training updates');ax.grid(alpha=.2);ax.set_ylim(0,75)
axs[0].set_ylabel('Validation Dice (%)');axs[-1].legend(frameon=False)
fig.suptitle('UNETR: matched MSD fold 0 / seed 0, no augmentation. Shading: bands warmup.')
fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
text=f"""# UNETR bands60 diagnosis

Both runs use the same initialization, 10 training/52 validation crops, no augmentation, 245 updates, optimizer/LR, precision and dataset/source hashes. Logged loss and Dice match exactly through updates {prefix}. The difference begins after bands activate at update51 and ramp to full weight by update100.

| Metric | Dice+CE | +bands60 |
|---|---:|---:|
"""
for key in ['validation_macro_dice_percent','training_macro_dice_percent','best_macro_dice_percent','validation_anterior_dice_percent','validation_posterior_dice_percent','binary_foreground_dice_percent','foreground_fn','foreground_fp','anterior_to_posterior','posterior_to_anterior']:
 text+=f"| {key} | {records['dice'][key]:.2f} | {records['dice_bands'][key]:.2f} |\n"
text+="""
The bands objective supervises the union of anterior and posterior against background. Its loss is invariant to exchanging anterior/posterior logits or labels, verified using the archived implementation. Thus better external foreground segmentation can coexist with worse anatomical classification. Observed binary foreground Dice rises, while A/P swaps increase and both class Dice scores fall. The degradation also occurs on training crops, so it is not solely validation overfitting.

Calibration achieved median 60% and p95 63.82%, below the 75% ceiling, on all ten training crops at update 50. The coefficient then remains fixed; this is not a guarantee of a 60% parameter-gradient contribution during later training. The learning curves initially improve with bands, then lose to baseline after about 184 updates. Best-checkpoint Dice is also lower, so final-checkpoint selection alone does not explain the outcome.

Interpretation: a plausible explanation is excessive emphasis on the external boundary before A/P discrimination is stable. The four requested 10/20% runs test strength while preserving all other settings. This single pair does not prove an architecture-wide weakness or establish an optimal weight.
"""
(out/'README.md').write_text(text)
print(json.dumps(dict(records=records,prefix=prefix,mechanism=mechanism),indent=2))
