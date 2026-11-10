"""Summarize existing validation errors for the proposed two-part input branch.

This read-only analysis of completed runs does not define training weights or
select training voxels. Geometry checks use annotation only as an audit target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.depth_supervision.geometry import longitudinal_plane, world_points


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'evaluation_output/depth_swinunetr_20261006/depth_error_audit_completed.json'
FOCAL = ROOT / 'evaluation_output/depth_weight_variants_20261006/final_focal_snapshot.json'
OUT = ROOT / 'evaluation_output/unwrapped_branch_design/evidence.json'


def main():
    audit = json.loads(SOURCE.read_text())
    cases = list(audit['full']['depth_vs_dice_patient_changes_pp'])
    counts = {}
    for case in cases:
        image = nib.load(ROOT / f'datasets/Dataset101_MSD/labelsTr/{case}.nii.gz')
        labels = np.asarray(image.dataobj)
        plane = longitudinal_plane(labels, image.affine)
        xyz = world_points(np.argwhere(labels > 0), image.affine)
        depth = np.abs((xyz - plane['center_mm']) @ plane['normal'])
        bins, sizes = np.unique(np.floor(depth / 2).astype(int), return_counts=True)
        for band, size in zip(bins, sizes):
            counts[str(band)] = counts.get(str(band), 0) + int(size)
    expected = {k: v['gt'] for k, v in audit['full']['depth']['bands'].items()}
    if counts != expected:
        raise ValueError('Native annotation depth counts disagree with the saved audit')

    results = {}
    for regime in ('5pct', 'full'):
        run = audit[regime]['depth']
        summary, bands = run['summary'], run['bands']
        for name in ('missed', 'extra', 'swaps'):
            if sum(b[name] for b in bands.values()) != summary[name]:
                raise ValueError(f'{regime}: inconsistent {name} total')
        total = sum(summary[k] for k in ('missed', 'extra', 'swaps'))
        gt = sum(b['gt'] for b in bands.values())
        results[regime] = {
            'macro_dice': summary['macro_dice'],
            'error_counts': {k: summary[k] for k in ('missed', 'extra', 'swaps')},
            'extra_fraction_of_errors': summary['extra'] / total,
            'swaps_fraction_at_0_to_6mm': sum(bands[str(k)]['swaps'] for k in range(3)) / summary['swaps'],
            'annotated_tissue_fraction_at_8mm_or_more': sum(v['gt'] for k, v in bands.items() if int(k) >= 4) / gt,
            'bands': [{
                'depth_range_mm': [2 * int(k), 2 * (int(k) + 1)],
                **{key: v[key] for key in ('gt', 'union', 'missed', 'extra', 'swaps', 'error_rate')},
            } for k, v in bands.items()],
        }
    focal = json.loads(FOCAL.read_text())['run']['completion']['validation']
    comparison = {key: focal[key] - audit['full']['depth']['summary'][key]
                  for key in ('missed', 'extra', 'swaps', 'macro_dice')}
    result = {
        'status': 'design evidence; no new model trained',
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (SOURCE, FOCAL)},
        'validation_cases': len(cases),
        'depth_definition': 'Absolute distance from annotation-derived longitudinal plane, 2 mm audit bands. Not distance from the tissue boundary. Training weights retain their existing signed bands.',
        'native_annotation_band_counts_verified': True,
        'aggregation': 'Pooled voxel counts over 52 validation cases; not mean per-case error rates.',
        'limitations': [
            'One fold and seed, already used for model selection; exploratory design evidence.',
            'Highest-depth bins have little annotated tissue; report denominators.',
            'These counts do not identify which errors are uncertain or confidently wrong.',
            'Validation labels must not become branch inputs or training sample selectors.',
            'Straightening checks do not establish the validity of two-part unwrapping.',
        ],
        'depth_baselines': results,
        'full_focal_minus_original_depth': comparison,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'output': str(OUT), 'verified_cases': len(cases),
        '5pct_extra_fraction': results['5pct']['extra_fraction_of_errors'],
        'full_swap_fraction_0_to_6mm': results['full']['swaps_fraction_at_0_to_6mm'],
        'full_focal_minus_original_depth': comparison,
    }, indent=2))


if __name__ == '__main__':
    main()
