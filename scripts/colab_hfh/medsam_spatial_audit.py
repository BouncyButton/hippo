"""Runtime utilities for the archived HFH protocol."""
import numpy as np

from scipy import ndimage as ndi

def spatial(truth, prediction):
    """Exact two-step six-connected BAND and separate physical EDT diagnostics."""
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    structure = ndi.generate_binary_structure(3, 1)
    inner = truth & ~ndi.binary_erosion(truth, structure, iterations=2)
    outer = ndi.binary_dilation(truth, structure, iterations=2) & ~truth
    regions = dict(inner=inner, core=truth & ~inner, outer=outer,
                   distant_background=~truth & ~outer)
    fp, fn = prediction & ~truth, truth & ~prediction
    l1 = ndi.distance_transform_cdt(~truth, metric='taxicab')
    mm = ndi.distance_transform_edt(~truth, sampling=(1., 1., 2.))
    components, count = ndi.label(prediction, structure)
    overlaps = np.unique(components[truth]); overlaps = overlaps[overlaps != 0]
    detached = prediction & ~np.isin(components, overlaps)
    empty = np.broadcast_to(~truth.any(axis=(0, 1))[None, None, :], truth.shape)
    row = dict(tp=int((truth & prediction).sum()), fp=int(fp.sum()), fn=int(fn.sum()),
               gt_voxels=int(truth.sum()), predicted_voxels=int(prediction.sum()),
               region_voxels={k: int(v.sum()) for k, v in regions.items()},
               errors_by_region={k: int(((fp | fn) & v).sum()) for k, v in regions.items()},
               fp_l1_steps={k: int((fp & m).sum()) for k, m in {
                   '1': l1 == 1, '2': l1 == 2, '3_4': (l1 > 2) & (l1 <= 4), 'over4': l1 > 4}.items()},
               fp_mm={k: int((fp & m).sum()) for k, m in {
                   'le1': mm <= 1, '1_2': (mm > 1) & (mm <= 2),
                   '2_4': (mm > 2) & (mm <= 4), 'over4': mm > 4}.items()},
               detached_component_fp=int((fp & detached).sum()),
               fp_on_empty_slices=int((fp & empty).sum()),
               predicted_components=int(count))
    row['dice'] = 2 * row['tp'] / max(2 * row['tp'] + row['fp'] + row['fn'], 1)
    return row, regions
