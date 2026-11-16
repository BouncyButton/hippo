"""CPU-only interpretation of saved MedSAM spatial-audit probability maps."""
import hashlib
import json
from pathlib import Path

import numpy as np


def dice(truth, prediction):
    return float(2 * (truth & prediction).sum() / max(int(truth.sum()+prediction.sum()), 1))


def run():
    from scripts.colab_hfh.medsam_recovery import copy_verified
    root = Path('/content/hippo_hfh')
    out = root/'medsam_spatial125_v1'
    remote = Path('/content/hfh_storage/results_medsam3_fixed125_v1/spatial_audit')
    analysis = json.loads((out/'analysis.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    records = []
    thresholds = np.linspace(.1, .9, 33)
    for r in analysis['records']:
        case = r['case']
        data = {}
        for arm in ('baseline', 'band'):
            path = out/(arm+'__'+case+'.npz')
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[path.name]:
                raise ValueError('Probability archive changed')
            with np.load(path) as z:
                data[arm] = {k:z[k] for k in z.files}
        a, b = data['baseline'], data['band']
        truth = a['truth'].astype(bool)
        empty = np.broadcast_to(~truth.any(axis=(0, 1))[None, None, :], truth.shape)
        new_fp = ~truth & ~a['prediction'].astype(bool) & b['prediction'].astype(bool)
        row = dict(case=case, split=r['split'], new_fp_on_empty_slices=int((new_fp & empty).sum()),
                   fields_on_new_fp_empty_slices={arm:{k:float(v[k][new_fp & empty].mean()) if (new_fp & empty).any() else None
                     for k in ('mask','score','presence','probability')} for arm,v in data.items()}, curves={})
        for arm, v in data.items():
            product = v['mask']*v['score']*v['presence']
            row[arm+'_reconstruction_disagreements'] = int(((product>.5) != v['prediction'].astype(bool)).sum())
            row['curves'][arm] = [dice(truth, v['probability']>t) for t in thresholds]
        # These are descriptive factor swaps between the per-pixel winning
        # instances, not actual model interventions or a deployable predictor.
        mixed = {'band_mask_baseline_confidence': b['mask']*a['score']*a['presence'],
                 'baseline_mask_band_confidence': a['mask']*b['score']*b['presence']}
        row['factor_swaps'] = {k:dict(dice=dice(truth, v>.5),
             fp=int(((v>.5)&~truth).sum()), fn=int(((v<=.5)&truth).sum()),
             fp_empty_slices=int(((v>.5)&empty).sum())) for k,v in mixed.items()}
        records.append(row)
    result = dict(optimizer_updates=0, inference_calls=0, records=records,
        caveat='Factor swaps are post-hoc descriptive, with potentially different winning instances; they do not establish gradient causality. Thresholds selected on training support only; not independent calibration data.',
        thresholds=thresholds.tolist(), summary={})
    for arm in ('baseline','band'):
        train = np.mean([r['curves'][arm] for r in records if r['split']=='training_support'], axis=0)
        index = int(np.argmax(train))
        result['summary'][arm] = dict(training_selected_threshold=float(thresholds[index]),
            training_dice=float(train[index]), validation_dice=float(np.mean([r['curves'][arm][index] for r in records if r['split']=='validation'])))
    for mode in ('band_mask_baseline_confidence','baseline_mask_band_confidence'):
        result['summary'][mode] = {split:{k:float(np.mean([r['factor_swaps'][mode][k] for r in records if r['split']==split])) if k=='dice'
            else sum(r['factor_swaps'][mode][k] for r in records if r['split']==split)
            for k in ('dice','fp','fn','fp_empty_slices')} for split in ('training_support','validation')}
    path = out/'confidence_followup.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    copy_verified(path, remote/path.name)
    print(json.dumps(result['summary']), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = sorted([r for r in analysis['records'] if r['split']=='validation'],
                   key=lambda r:r['band']['dice']-r['baseline']['dice'])[:2]
    fig, axes = plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,r in enumerate(cases):
        case=r['case']
        with np.load(out/('baseline__'+case+'.npz')) as a, np.load(out/('band__'+case+'.npz')) as b, np.load(root/'bundle/data'/(case+'.npz')) as image:
            truth=a['truth'].astype(bool);pa=a['prediction'].astype(bool);pb=b['prediction'].astype(bool)
            new_fp=~truth&~pa&pb
            z=int(new_fp.sum(axis=(0,1)).argmax())
            grey=image['image'].transpose(0,2,1)[:,:,z].T
            lo,hi=np.percentile(grey,[1,99])
            for col,pred in enumerate((pa,pb)):
                ax=axes[row,col];ax.imshow(grey,cmap='gray',vmin=lo,vmax=hi)
                rgba=np.zeros((*grey.shape,4));fp=(pred&~truth)[:,:,z].T;fn=(~pred&truth)[:,:,z].T
                rgba[fp]=[1,.15,.15,.85];rgba[fn]=[0,.8,1,.85]
                ax.imshow(rgba);ax.set_title(('Baseline','BAND')[col]+' | '+case.rsplit('_',2)[-2]+'_'+case.rsplit('_',1)[-1])
                if truth[:,:,z].any():ax.contour(truth[:,:,z].T,levels=[.5],colors=['lime'],linewidths=.8)
            ax=axes[row,2];ax.imshow(grey,cmap='gray',vmin=lo,vmax=hi)
            rgba=np.zeros((*grey.shape,4));rgba[new_fp[:,:,z].T]=[1,.8,0,.9];ax.imshow(rgba)
            if truth[:,:,z].any():ax.contour(truth[:,:,z].T,levels=[.5],colors=['lime'],linewidths=.8)
            ax.set_title(f'New false positives | slice {z}, GT={int(truth[:,:,z].sum())}')
            for ax in axes[row]:ax.axis('off')
    fig.suptitle('Saved125 checkpoints: red=false positive, cyan=miss, green=true contour\nYellow=new false positive; slice selected by largest new-FP count')
    path=out/'spatial_examples.png';fig.savefig(path,dpi=140);plt.close(fig)
    copy_verified(path,remote/path.name)
    print('CPU_CONFIDENCE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    run()
