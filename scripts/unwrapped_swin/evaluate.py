"""Native-space evaluation with chart features, keeping the original metrics."""
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from scripts.depth_supervision.train import autocast

@torch.no_grad()
def evaluate(model, patients, device, *, geometry_cache=None, atlas_cache=None, detailed=False, predictions_path=None):
    model.eval()
    rows = []
    exported = {}
    for item in patients.items:
        with autocast(device):
            logits = model(item['image'][None].to(device), atlas_cache.get(item) if atlas_cache else None)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite validation output')
        prediction = logits.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        target = item['label'][0].numpy().astype(np.uint8)
        row = {'case_name': item['case_name']}
        for c, label in ((1, 'anterior'), (2, 'posterior'), (None, 'foreground')):
            g = target > 0 if c is None else target == c
            p = prediction > 0 if c is None else prediction == c
            denominator = int(g.sum() + p.sum())
            row[label + '_dice'] = float(2 * (g & p).sum() / denominator) if denominator else 1.
        row['macro_dice'] = (row['anterior_dice'] + row['posterior_dice']) / 2
        fg_g, fg_p = target > 0, prediction > 0
        row.update(missed=int((fg_g & ~fg_p).sum()), extra=int((fg_p & ~fg_g).sum()),
                   swaps=int(((target != prediction) & fg_g & fg_p).sum()))
        if detailed:
            spacing = np.linalg.norm(item['affine'][:3, :3], axis=0)
            structure = ndimage.generate_binary_structure(3, 1)
            sg = fg_g & ~ndimage.binary_erosion(fg_g, structure=structure)
            sp = fg_p & ~ndimage.binary_erosion(fg_p, structure=structure)
            if sp.any() and sg.any():
                distances = np.r_[ndimage.distance_transform_edt(~sg, sampling=spacing)[sp],
                                   ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]]
                row['surface_distance_mean_mm'] = float(distances.mean())
                row['hd95_mm'] = float(np.quantile(distances, .95))
            else:
                row['surface_distance_mean_mm'] = row['hd95_mm'] = None
            geometry = geometry_cache.get(item, fields=False)
            row['depth_bands'] = []
            union = fg_g | fg_p
            for b in np.unique(geometry['bin_ids'][union]):
                mask = geometry['bin_ids'] == b
                row['depth_bands'].append(dict(signed_bin=int(b), union=int((mask & union).sum()),
                    missed=int((mask & fg_g & ~fg_p).sum()), extra=int((mask & fg_p & ~fg_g).sum()),
                    swaps=int((mask & (target != prediction) & fg_g & fg_p).sum())))
        rows.append(row)
        if predictions_path:
            exported[item['case_name']] = prediction
    summary = {key: float(np.mean([r[key] for r in rows]))
               for key in ('anterior_dice', 'posterior_dice', 'foreground_dice', 'macro_dice')}
    summary.update({key: sum(r[key] for r in rows) for key in ('missed', 'extra', 'swaps')})
    if detailed:
        for key in ('surface_distance_mean_mm', 'hd95_mm'):
            valid = [r[key] for r in rows if r[key] is not None]
            summary[key] = float(np.mean(valid)) if valid else None
        summary['undefined_surface_cases'] = sum(r['hd95_mm'] is None for r in rows)
    if predictions_path:
        with Path(predictions_path).open('wb') as stream:
            np.savez_compressed(stream, **exported)
    return {'summary': summary, 'cases': rows}

