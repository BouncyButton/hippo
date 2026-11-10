"""Read saved predictions; emit pooled counts without individual records."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path('/mnt/beegfsstudents/home/3160552/band_chamfer_fold0_seed0_5pct_20261008_01')
RUNS = ('bands', 'pair13', 'pair_independent13', 'pair13_plus_sdf8')
BINS = np.array([0., 1., 2., 3., 5., np.inf])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def surface(y, valid):
    points, classes, anchors, directions = [], [], [], []
    for axis in range(3):
        lo, hi = [slice(None)]*3, [slice(None)]*3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = y[tuple(lo)], y[tuple(hi)]
        cross = ((a > 0) != (b > 0)) & valid[tuple(lo)] & valid[tuple(hi)]
        coords = np.argwhere(cross)
        step = np.eye(3, dtype=int)[axis]
        a_inside = a[cross] > 0
        inside = coords + (~a_inside)[:, None]*step
        anchors.append(inside)
        directions.append((2*a_inside.astype(int)-1)[:, None]*step)
        classes.append(y[tuple(inside.T)])
        points.append(coords + .5*step)
    return tuple(np.concatenate(x) for x in (points, classes, anchors, directions))


def hist(distance, mask):
    return np.histogram(distance[mask], BINS)[0].tolist()


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    results = json.loads((ROOT/'results.json').read_text())
    expected = {r['objective']: r for r in results['rows']}
    data = pd.read_pickle(protocol['pkl'], compression='gzip')
    labels = {str(row.subject_id): np.asarray(row.label_data) for row in data.itertuples()}
    del data
    predictions, hashes = {}, {}
    for name in RUNS:
        folder = ROOT/'runs'/name
        path = folder/'validation_predictions.npz'
        hashes[name] = sha(path)
        record = json.loads((folder/'completion.json').read_text())
        assert hashes[name] == record['artifacts'][path.name]
        with np.load(path) as archive:
            predictions[name] = {key: archive[key] for key in archive.files}
        assert set(predictions[name]) == set(protocol['splits']['5pct']['val'])
    cases, support = {r: [] for r in RUNS}, []
    for split in ('val', 'train'):
        for name in protocol['splits']['5pct'][split]:
            native = labels[name].astype(np.uint8)
            meta = protocol['native_geometry'][name]
            assert list(native.shape) == meta['native_shape']
            assert hashlib.sha256(native.tobytes()).hexdigest() == meta['label_array_sha256']
            spacing = np.linalg.norm(np.asarray(meta['model_affine'])[:3, :3], axis=0)
            assert np.allclose(spacing, 1.)
            before = (64-np.asarray(native.shape))//2
            after = 64-np.asarray(native.shape)-before
            padding = tuple(zip(before, after))
            y = np.pad(native, padding)
            valid = np.pad(np.ones(native.shape, bool), padding)
            points, classes, anchors, directions = surface(y, valid)
            if split == 'train':
                counts = {}
                for separation in (1, 3, 5, 7):
                    n = (separation-1)//2
                    okay = np.ones(len(anchors), bool)
                    for offset in range(-n, n+2):
                        coords = anchors+offset*directions
                        okay &= ((coords >= 0) & (coords < 64)).all(1)
                        coords = np.clip(coords, 0, 63)
                        okay &= valid[tuple(coords.T)] & ((y[tuple(coords.T)] > 0) == (offset <= 0))
                    counts[str(separation)] = int(okay.sum())
                support.append(dict(case_name=name, counts=counts))
                continue
            tree = cKDTree(points)
            coords = np.argwhere(valid)
            dist, nearest = tree.query(coords, workers=1)
            distance = np.full(y.shape, np.nan)
            distance[valid] = dist
            nearest_class = np.zeros_like(y)
            nearest_class[valid] = classes[nearest]
            for run in RUNS:
                pred = predictions[run][name]
                assert pred.shape == y.shape and np.isin(pred, [0, 1, 2]).all()
                fp = (pred > 0) & (y == 0)
                fn = (pred == 0) & (y > 0)
                swaps = (pred > 0) & (y > 0) & (pred != y)
                row = dict(case_name=name, extra=int(fp.sum()), missed=int(fn.sum()),
                           swaps=int(swaps.sum()), padding_extra=int((fp & ~valid).sum()),
                           extra_by_distance=hist(distance, fp & valid),
                           missed_by_distance=hist(distance, fn),
                           background_by_distance=hist(distance, (y == 0) & valid),
                           foreground_by_distance=hist(distance, y > 0),
                           extra_by_nearest_gt_class=[int((fp & valid & (nearest_class == c)).sum()) for c in (1, 2)],
                           missed_by_gt_class=[int((fn & (y == c)).sum()) for c in (1, 2)])
                if run == 'pair_independent13':
                    old = predictions['pair13'][name]
                    row['extra_removed_vs_pair13'] = hist(distance, (old > 0) & (pred == 0) & (y == 0) & valid)
                    row['extra_added_vs_pair13'] = hist(distance, (old == 0) & (pred > 0) & (y == 0) & valid)
                    row['missed_added_vs_pair13'] = hist(distance, (old > 0) & (pred == 0) & (y > 0))
                    row['missed_rescued_vs_pair13'] = hist(distance, (old == 0) & (pred > 0) & (y > 0))
                cases[run].append(row)
    summaries = {}
    for run, rows in cases.items():
        total = {k: np.asarray([r[k] for r in rows]).sum(axis=0).tolist()
                 for k in rows[0] if k != 'case_name'}
        for key in ('extra', 'missed', 'swaps'):
            assert total[key] == expected[run][key], (run, key, total[key], expected[run][key])
        for tag in ('extra', 'missed'):
            fractions = [sum(r[tag+'_by_distance'][:2])/r[tag] for r in rows if r[tag]]
            total[tag+'_within2mm_case_fraction_median'] = float(np.median(fractions))
        summaries[run] = total
    pooled_support = {str(d): sum(row['counts'][str(d)] for row in support)
                      for d in (1, 3, 5, 7)}
    out = dict(protocol_identity=protocol['identity_sha'], validation_case_count=len(cases[RUNS[0]]),
               distance_definition='Euclidean distance from voxel center to nearest annotated outer axis-edge midpoint, native 1 mm FOV; bins [0,1), [1,2), [2,3), [3,5), [5,infinity). Padding errors reported separately.',
               interpretation='Post-hoc validation error analysis for hypothesis generation; not independent confirmation. No loss coefficients fitted from validation.',
               summaries=summaries, training_case_count=len(support),
               training_pair_support_pooled=pooled_support)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
