"""Training-label-only spatial prior; threshold selected without validation labels."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'geofwi3d_colab_seed_search_20261018/bundle'
manifest=json.loads((BUNDLE/'data/fold_0_manifest.json').read_text())
split=json.loads((BUNDLE/'splits.json').read_text())[0]
def mask(name):return np.asarray(nib.load(BUNDLE/'data'/manifest['cases'][name]['label']).dataobj,dtype=np.uint8)
train=np.stack([mask(c) for c in split['train']]);counts=train.sum(0,dtype=np.uint8)
scores=[]
for k in range(1,19):
    dice=[]
    for y in train:
        prediction=(counts-y)>=k
        dice.append(2*np.count_nonzero(prediction&y.astype(bool))/(prediction.sum()+y.sum()))
    scores.append(float(np.mean(dice)))
best_k=int(np.argmax(scores))+1
cutoff=best_k/18
predictions={'majority_vote':counts>=10,'training_loo_selected':counts/19>=cutoff}
result={}
for method,pred in predictions.items():
    values=[]
    for name in split['val']:
        y=mask(name);values.append(2*np.count_nonzero(pred&y.astype(bool))/(pred.sum()+y.sum()))
    result[method]=dict(validation_mean_dice=float(np.mean(values)),
        predicted_fraction=float(pred.mean()),cases_above_50=int(np.sum(np.array(values)>.5)))
report=dict(method='One fixed spatial mask for every case, learned from training labels only; no image inputs, no GPU, no model training',
    threshold_selection='Leave-one-training-volume-out Dice, with ties resolved toward the lower k; no validation threshold search',
    selected_threshold=cutoff,selected_training_loo_dice=max(scores),training_loo_scores=scores,
    results=result,interpretation='Diagnostic reference only. Does not prove what spatial features UNETR learned.')
(HERE/'spatial_prior_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
