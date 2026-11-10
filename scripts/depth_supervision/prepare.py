"""Freeze the requested protocol, original patient subsets, and image geometry."""
from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .geometry import GeometryConfig
from .losses import ARMS
from .utils import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'docs/experiments/depth_swinunetr_20261006'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--production-pkl-sha256', required=True)
    parser.add_argument('--verified-uncompressed-sha256', required=True)
    args = parser.parse_args()
    pkl = ROOT / 'datasets/Dataset101_MSD/msd_hippocampus_full.pkl'
    uncompressed = hashlib.sha256(gzip.open(pkl, 'rb').read()).hexdigest()
    if uncompressed != args.verified_uncompressed_sha256:
        raise ValueError('Local pickle differs from the verified cluster contents')
    if len(args.production_pkl_sha256) != 64:
        raise ValueError('Production archive digest must be SHA-256')
    full_path = ROOT / 'datasets/Dataset101_MSD/splits_final.json'
    low_path = ROOT / 'experiments/compute_matched_baselines_20261003/splits_10patients.json'
    full = json.loads(full_path.read_text())[0]
    low = json.loads(low_path.read_text())[0]
    assert len(full['train']) == 208 and len(full['val']) == 52 and len(low['train']) == 10
    assert set(low['train']) <= set(full['train']) and set(low['val']) == set(full['val'])
    assert not set(full['train']) & set(full['val'])
    dataframe = pd.read_pickle(pkl, compression='gzip')
    metadata = {}
    for row in dataframe.itertuples(index=False):
        name = str(row.subject_id)
        native = nib.load(str(ROOT / 'datasets/Dataset101_MSD/labelsTr' / (name + '.nii.gz')))
        labels = np.asarray(native.dataobj)
        if not np.array_equal(labels, row.label_data):
            raise ValueError('Native label axes do not match pickle: ' + name)
        spacing = np.linalg.norm(native.affine[:3, :3], axis=0)
        if not np.allclose(spacing, 1., atol=1e-6):
            raise ValueError('Existing 3D SDF comparison assumes the MSD 1-mm grid')
        shape = np.array(labels.shape)
        pad = np.maximum(64 - shape, 0) // 2
        crop = np.maximum(shape - 64, 0) // 2
        affine = native.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ (crop-pad)
        metadata[name] = dict(native_shape=list(labels.shape), model_affine=affine.tolist(),
                              label_array_sha256=hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest())
    sources = sorted((ROOT / 'scripts/depth_supervision').glob('*.py'))
    sources += [ROOT / 'baselines/swin_unetr/swin_unetr.py', ROOT / 'scripts/depth_supervision/run.sh']
    protocol = dict(name='depth_swinunetr_fold0_seed0_20261006', dataset='MSD Task04 / Dataset101_MSD',
        fold=0, seed=0, max_epochs=75, arms=list(ARMS), baseline_loss='dice',
        model='MONAI SwinUNETR', model_parameters=15703029, feature_size=24, spatial_size=[64,64,64],
        batch_size=1, drop_rate=0., activation_checkpointing=False, pretrained=False,
        training_augmentation='none', precision='CUDA bfloat16 autocast; FP32 loss and optimizer',
        learning_rate=1e-4, weight_decay=1e-5, step_size=20, lr_gamma=.5,
        early_stopping=dict(patience=8, min_epochs=25, min_delta=.0005),
        selection='maximum mean per-patient anterior/posterior hard Dice, evaluated each epoch',
        geometry=asdict(GeometryConfig()),
        depth_objective='multiclass CE weighted by equal signed-depth-band means in GT foreground + 8-mm background margin; sparse band mass min(N/64,1)',
        sdf_objective='mean across A/P of (p-y)*phi, with the same depth voxel weights in both SDF arms; normalized clipped distances +/-8 mm',
        layer_sdf='1-mm physical cross-sections parallel to the plane; nearest-label rasterization, per-class 2D EDT, interpolated absolute distances, original native label signs',
        plane='foreground centroid + longest physical PCA axis; normal is RAS right projected orthogonally to longitudinal axis; keep signed sides separate',
        calibration=dict(prefix_epochs=5, ramp_epochs=5, target_ratio=.1, cap_ratio=.5,
                         max_cases=32, seed=0, source='own epoch-5 Dice-only state, training patients only'),
        permitted_auxiliaries=['depth-balanced multiclass CE', 'classwise 3D SDF', 'classwise cross-sectional SDF'],
        required_monai='1.6.0', splits={'full': full, '5pct': low},
        low_data_fraction_actual=10/208,
        pkl_sha256=args.production_pkl_sha256, local_pkl_sha256=digest(pkl),
        uncompressed_pickle_sha256=uncompressed,
        full_split_sha256=digest(full_path), low_split_sha256=digest(low_path),
        native_geometry=metadata,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources})
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_json(DIRECTORY / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=str(DIRECTORY/'protocol.json'), pkl_sha256=protocol['pkl_sha256'],
                         full_train=208, low_train=10, validation=52, source_files=len(sources))))


if __name__ == '__main__':
    main()
