"""Read-only spatial probes; labels in validation never affect training."""
from pathlib import Path
import random
import time
import numpy as np
from scipy import ndimage as ndi
import torch
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import adaptive_bands, objectives
from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss


def spatial(y, p, original, adaptive):
    foreground = y > 0
    masks = dict(fn=foreground & (p == 0), fp=~foreground & (p > 0),
                 internal_swaps=foreground & (p > 0) & (y != p))
    distance = np.where(foreground, ndi.distance_transform_cdt(foreground, metric='taxicab'),
                        ndi.distance_transform_cdt(~foreground, metric='taxicab'))
    rows = {}
    for key, mask in masks.items():
        rows[key] = dict(total=int(mask.sum()), original_band=int((mask & original).sum()),
                         adaptive_band=int((mask & adaptive).sum()),
                         distance_bins={str(d):int((mask & (distance==d)).sum()) for d in (1,2,3,4)})
        rows[key]['distance_bins']['>4'] = int((mask & (distance > 4)).sum())
    def components(mask, intersect):
        labels, count = ndi.label(mask)
        sizes = np.bincount(labels.ravel())
        overlap = set(np.unique(labels[intersect]))
        missed = [int(sizes[i]) for i in range(1,count+1) if i not in overlap]
        return dict(components=int(count), disconnected_count=len(missed), disconnected_voxels=sum(missed))
    return dict(errors=rows, support_voxels=dict(original=int(original.sum()),adaptive=int(adaptive.sum()),
                foreground=int(foreground.sum()),volume=int(y.size)),
                predicted_components=components(p > 0,foreground),
                true_components=components(foreground,p > 0))


def gradient_probe(raw, label, identity, criterion, variant, coefficient):
    detached = [v.detach().float() for v in raw] if isinstance(raw,(tuple,list)) else raw.detach().float()
    z = detached[0] if isinstance(detached,list) else detached
    z.requires_grad_(True)
    gb = torch.autograd.grad(objectives.base_loss(detached,label,identity,criterion),z)[0]
    result={}
    for name in ('original_bands','weighted','adaptive'):
        value=(OuterBoundaryBandLoss(foreground_class_ids=(1,2,3),complement_class_ids=(0,))(z,label).loss
               if name=='original_bands' else adaptive_bands.loss(z,label,name))
        g = torch.autograd.grad(value,z)[0]
        errors=(label>0)^(z.detach().argmax(1,keepdim=True)>0)
        energy=g.square().sum(1,keepdim=True)
        total=float(energy.sum()); base=float(gb.square().sum())
        error_energy=float(energy[errors].sum())
        result[name]=dict(raw_norm_ratio=(total/max(base,1e-30))**.5,
            error_gradient_energy=error_energy,correct_gradient_energy=total-error_energy,
            fraction_energy_on_external_errors=error_energy/max(total,1e-30),
            weighted_norm_ratio=coefficient*(total/max(base,1e-30))**.5 if name==variant else None)
    return result


def overlay(path, item, prediction, original, adaptive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y=item['label'][0].numpy(); im=item['image'][0].numpy()
    index=int((y>0).sum((1,2)).argmax())
    region=(y>0)|(prediction>0)|original|adaptive
    coordinates=np.argwhere(region)
    lo=np.maximum(coordinates.min(0)-2,0); hi=np.minimum(coordinates.max(0)+3,y.shape)
    slices=(slice(lo[1],hi[1]),slice(lo[2],hi[2]))
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    truth=y[index][slices]; pred=prediction[index][slices]
    for ax in axes: ax.imshow(im[index][slices].T,cmap='gray',origin='lower'); ax.axis('off')
    axes[0].imshow(np.ma.masked_where(truth.T==0,truth.T),vmin=0,vmax=3,cmap='viridis',alpha=.7,origin='lower')
    axes[1].imshow(np.ma.masked_where(pred.T==0,pred.T),vmin=0,vmax=3,cmap='viridis',alpha=.7,origin='lower')
    error=(truth>0)^(pred>0)
    for ax,support in zip(axes[2:],(original,adaptive)):
        ax.imshow(np.ma.masked_where(~support[index][slices].T,support[index][slices].T),cmap='winter',vmin=0,vmax=1,alpha=.4,origin='lower')
        ax.imshow(np.ma.masked_where(~error.T,error.T),cmap='autumn',vmin=0,vmax=1,alpha=.85,origin='lower')
    for ax,title in zip(axes,('Ground truth','Prediction','Original bands + errors','Adaptive bands + errors')): ax.set_title(title)
    fig.suptitle(f"{item['case_name']} — array slice {index}; yellow external errors, cyan support")
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def capture(model, train, validation, device, amp, args, protocol, charged, flops, maximum, criterion):
    output=args.output; budget=protocol['budget_flops']
    quarter=4 if charged==maximum else next((q for q in (1,2,3) if flops>=budget*q/4 and not (output/f'spatial_{q*25:03d}.json').exists()),None)
    if quarter is None or (output/f'spatial_{quarter*25:03d}.json').exists(): return
    started=time.monotonic(); identity=protocol['model_identity']; variant=protocol['adaptive_variant']
    stage=output.parent.parent; cache=stage/'diagnostic_cache'/protocol.get('diagnostic_cache_group',identity)
    paired_predictions=protocol.get('paired_baseline_predictions',True)
    if paired_predictions: cache.mkdir(parents=True,exist_ok=True)
    from scripts.cross_dataset_campaign.capped_core import model_digest
    before=model_digest(model); mode=model.training
    cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all() if amp else []
    numpy_state=np.random.get_state(); python_state=random.getstate()
    rows=[]; cache_files=[]
    calibration=read(output/'calibration.json') if (output/'calibration.json').exists() else {}
    coefficient=calibration.get('weights',{}).get('bands',0.)
    try:
        model.eval()
        for split,patients in (('training',train),('validation',validation)):
            for index,item in enumerate(patients):
                image=item['image'][None].to(device); label=item['label'][None].to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                    raw=model(image)
                z=raw[0] if isinstance(raw,(list,tuple)) else raw
                i,o,_,_=adaptive_bands.masks(z.float(),label,'weighted')
                a,b,_,_=adaptive_bands.masks(z.float(),label,'adaptive')
                original=(i|o)[0,0].cpu().numpy(); expanded=(a|b)[0,0].cpu().numpy()
                y=label[0,0].cpu().numpy(); p=z.argmax(1)[0].cpu().numpy().astype(np.uint8)
                row=dict(split=split,case_name=item['case_name'],**spatial(y,p,original,expanded))
                reference=cache/f'{quarter*25:03d}_{split}_{index:02d}.npz'
                if variant=='baseline' and paired_predictions:
                    np.savez_compressed(reference,prediction=p,case_name=item['case_name'])
                    cache_files.append(dict(name=reference.name,sha256=digest(reference),bytes=reference.stat().st_size))
                elif paired_predictions:
                    with np.load(reference,allow_pickle=False) as data:
                        assert str(data['case_name'])==item['case_name']; old=data['prediction']
                    row['paired_baseline']=dict(corrected_voxels=int(((old!=y)&(p==y)).sum()),
                        introduced_voxels=int(((old==y)&(p!=y)).sum()),
                        corrected_external=int((((old>0)^(y>0))&~((p>0)^(y>0))).sum()),
                        introduced_external=int((~((old>0)^(y>0))&((p>0)^(y>0))).sum()))
                else:
                    row['paired_baseline_unavailable']='Baseline prediction cache disabled for this comparison; compare archived aggregate metrics, not voxel transitions.'
                if split=='training':
                    row['gradient_probe']=gradient_probe(raw,label,identity,criterion,variant,coefficient)
                if quarter==4 and split=='validation' and index==0:
                    overlay(output/'band_prediction_overlay.png',item,p,original,expanded)
                rows.append(row)
        assert model_digest(model)==before,'Diagnostic probe mutated model'
        write(output/f'spatial_{quarter*25:03d}.json',dict(status='complete',quarter=quarter,
            model=identity,variant=variant,charged_updates=charged,optimization_flops=flops,
            actual_fraction=flops/budget,model_sha256=before,cases=rows,baseline_cache=cache_files,
            elapsed_seconds=time.monotonic()-started,diagnostic_forwards=len(rows),
            detached_logit_gradient_probes=4*len(train),optimizer_updates=0,
            scope='Read-only evaluation outside optimization budget. Training-only gradient probes; validation errors never enter optimization. Distances are six-connected voxel steps to opposite GT union, not millimetres. Adaptive validation support is a retrospective hypothetical coverage diagnostic.'))
    finally:
        model.train(mode); torch.set_rng_state(cpu)
        if amp: torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy_state); random.setstate(python_state)
