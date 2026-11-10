"""Training-only A/P geometry audit and non-image dataset fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.depth_supervision.train import Patients
from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe


def fingerprint(value):
    array = np.asarray(value, dtype=np.float32)
    return dict(shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def audit(protocol, dataset):
    dataframe = _load_pkl_dataframe(dataset)
    names = protocol['splits']['5pct']['train']
    patients = Patients(dataframe, names, protocol['native_geometry'])
    indexed = {str(row.subject_id): row for row in dataframe.itertuples(index=False)}
    records = []
    for item in patients.items:
        y = item['label'][0].numpy()
        anterior = (y == 1).sum((0, 2)); posterior = (y == 2).sum((0, 2))
        errors = np.cumsum(anterior)[:-1]+np.cumsum(posterior[::-1])[::-1][1:]
        faces = []
        for axis in range(3):
            lo, hi = [slice(None)]*3, [slice(None)]*3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            left, right = y[tuple(lo)], y[tuple(hi)]
            faces.append(int((((left == 1) & (right == 2)) | ((left == 2) & (right == 1))).sum()))
        row = indexed[item['case_name']]
        records.append(dict(case_name=item['case_name'], best_plane_first_anterior_y=int(errors.argmin()+1),
            best_plane_disagreement_voxels=int(errors.min()), foreground_voxels=int((y > 0).sum()),
            interface_faces_xyz=faces, mixed_slices=int(((anterior > 0) & (posterior > 0)).sum()),
            image_fingerprint=fingerprint(row.image_data), label_fingerprint=fingerprint(row.label_data),
            y_anterior_counts=anterior.tolist(), y_posterior_counts=posterior.tolist()))
    return dict(scope='Only the frozen 10 training cases; no validation-based geometry choices',
        cases=records, cases_checked=len(records), planar_cases=sum(r['best_plane_disagreement_voxels'] == 0 for r in records),
        nonplanar_cases=sum(r['best_plane_disagreement_voxels'] > 0 for r in records),
        orientation='RAS, spatial axis 1, anterior high Y; verified from existing native metadata and label counts'), patients


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    result, patients = audit(p, args.pkl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, axes = plt.subplots(2, 5, figsize=(17, 9), constrained_layout=True)
        for ax, item, row in zip(axes.flat, patients.items, result['cases']):
            y = item['label'][0].numpy()
            x = int(np.argwhere(y > 0)[:, 0].mean())
            label = y[x].T
            image = item['image'][0, x].numpy().T
            ax.imshow(image, cmap='gray', origin='lower')
            ax.imshow(np.ma.masked_equal(label, 0), cmap=ListedColormap(['black', '#ff705d', '#49a1e8']),
                      vmin=0, vmax=2, alpha=.55, origin='lower', interpolation='nearest')
            ax.axvline(row['best_plane_first_anterior_y']-.5, color='yellow', lw=.7)
            ax.set_title(row['case_name']+'\nPlane disagreements: '+str(row['best_plane_disagreement_voxels']), fontsize=9)
            ax.set_xlabel('Posterior ← Y → Anterior'); ax.set_ylabel('Z')
        fig.suptitle('Frozen 5% training cases: red anterior, blue posterior; yellow best-fit plane\nOriginal mixed-slice annotations are retained by the new losses')
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130); plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
