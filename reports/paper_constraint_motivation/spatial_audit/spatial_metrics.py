"""Spatial error counts for a prediction on the observed native crop."""
import numpy as np
from scipy import ndimage as ndi
from scripts.depth_supervision.geometry import longitudinal_plane, signed_distance

FACE = ndi.generate_binary_structure(3, 1)
NEIGHBORS = FACE.astype('uint8')
NEIGHBORS[1, 1, 1] = 0

def counts(y, pred, affine, classes):
    fg, guess = y > 0, pred > 0
    inside = fg & ~ndi.binary_erosion(fg, FACE, iterations=2, border_value=1)
    outside = ~fg & ndi.binary_dilation(fg, FACE, iterations=2)
    band = inside | outside
    wrong = fg != guess
    mismatch = y != pred
    r = dict(voxels=y.size, mismatches=int(mismatch.sum()), union_errors=int(wrong.sum()),
             band_union_errors=int((band & wrong).sum()), band_voxels=int(band.sum()))
    eligible = ndi.binary_erosion(np.ones_like(fg), FACE)
    nwrong = ndi.convolve(wrong.astype('uint8'), NEIGHBORS, mode='constant')
    centers = eligible & band & wrong
    r.update(centers=int(centers.sum()), isolated=int((centers & (nwrong == 0)).sum()),
             majority_correct=int((centers & (nwrong <= 2)).sum()))
    pairs, breaks = 0, 0
    for axis in range(3):
        a = [slice(None)] * 3
        b = a.copy()
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        ok = band[a] & band[b] & (fg[a] == fg[b])
        pairs += int(ok.sum())
        breaks += int((ok & (guess[a] != guess[b])).sum())
    r.update(faces=pairs, broken=breaks)
    plane = longitudinal_plane(y, affine)
    grid = np.ogrid[tuple(slice(0, n) for n in y.shape)]
    coeff = affine[:3, :3].T @ plane['normal']
    depth = sum(c*g for c,g in zip(coeff,grid)) + (affine[:3, 3]-plane['center_mm']) @ plane['normal']
    layers = np.floor(depth/2).astype('int16')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    roi = ndi.distance_transform_edt(~fg, sampling=spacing) <= 8
    # Rank layers by the same selected-voxel population as the implemented depth loss.
    occupied, size = np.unique(layers[roi], return_counts=True)
    intersect = np.isin(occupied, np.unique(layers[band]))
    occupied, size = occupied[intersect], size[intersect]
    order = occupied[np.lexsort((occupied, size))]
    low = band & np.isin(layers, order[:len(order)//2])
    high = band & ~low
    r.update(low=int(low.sum()), high=int(high.sum()), lowerr=int((wrong & low).sum()), higherr=int((wrong & high).sum()))
    dice = []
    class_errors = far = far4 = 0
    for c in range(1, classes):
        actual, chosen = y == c, pred == c
        assert actual.any(), 'no reference for foreground class'
        distance = abs(signed_distance(actual, tuple(spacing), 8.))
        errs = actual != chosen
        class_errors += int(errs.sum())
        far += int((errs & (distance > 2)).sum())
        far4 += int((errs & (distance > 4)).sum())
        dice.append(2*(actual & chosen).sum()/(actual.sum()+chosen.sum()))
    r.update(class_errors=class_errors, far=far, far4=far4, dice=np.mean(dice))
    assert r['lowerr']+r['higherr']==r['band_union_errors'] and r['broken']<=pairs
    return r
