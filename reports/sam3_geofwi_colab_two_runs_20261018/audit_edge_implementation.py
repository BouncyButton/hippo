"""Reproduce the completed-run implementation audit without GPU work or training."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    rows = REPORT/'completed_results'
    edge_dir = next(rows.glob('*dice_bands_edge*'))
    calibration = read(edge_dir/'calibration.json')
    history = read(edge_dir/'training_history.json')
    wb, we = (calibration['weights'][name] for name in ('bands', 'edge'))
    cosines, band_cosines, ratios = [], [], []
    for row in calibration['rows']:
        gram = np.asarray(row['gram'])
        base = np.array([1., wb]) @ gram[:2,:2] @ np.array([1., wb])
        ratios.append(float(we*np.sqrt(gram[2,2]/base)))
        cosines.append(float((gram[0,2]+wb*gram[1,2])/np.sqrt(base*gram[2,2])))
        band_cosines.append(float(gram[1,2]/np.sqrt(gram[1,1]*gram[2,2])))
    # Run the current MedSAM3 MSD coefficient selector against the exact saved SAM3 Gram rows.
    source = ROOT/'scripts/colab_medsam3/engine.py'
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'weights_for')
    scope = {'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    padded = []
    for row in calibration['rows']:
        gram = np.zeros((5,5)); gram[:3,:3] = row['gram']
        padded.append(dict(gram=gram.tolist()))
    med = scope['weights_for']('bands_edge',padded)
    for name, weight in calibration['weights'].items():
        np.testing.assert_allclose(med['weights'][name],weight,rtol=1e-12,atol=1e-12)
    files = ['scripts/medsam3_supervised_constraints.py',
             'thesis/new_constraints/edge_consistency.py',
             'thesis/new_constraints/bands/outer_boundary.py']
    frozen_med = ROOT/'reports/medsam3_msd_colab_matched_3fold_3seed_noaug_20261015/bundle'
    hashes = {}
    for name in files:
        paths = [ROOT/name, REPORT/'bundle/source'/name, frozen_med/'source'/name]
        values = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert len(set(values)) == 1
        hashes[name] = values[0]
    scalar = [{'update':r['update'],'native':r['native'],
               'scaled_bands':r['ramp']*wb*r['bands'],
               'scaled_edge':r['ramp']*we*r['edge'],
               'scaled_edge_over_native':r['ramp']*we*r['edge']/r['native']}
              for r in history]
    med_weights = {p.stem:read(p)['edge']['weight'] for p in sorted((frozen_med/'calibration').glob('fold*.json'))}
    result = dict(
        scope='Read-only scientific audit; no new training, evaluation, or model changes.',
        same_loss_source_sha256=hashes,
        estimator_verification='4 tests passed: full-volume values and gradients, no wraparound, confidence gradients, extreme logits.',
        medsam3_selector_matches_sam3_weights=True,
        calibrated_weights=calibration['weights'],
        calibrated_statistics=calibration['statistics'],
        edge_gradient_ratio=dict(min=min(ratios),median=float(np.median(ratios)),
                                 p95=float(np.quantile(ratios,.95)),max=max(ratios)),
        cosine_edge_native_plus_bands=dict(min=min(cosines),median=float(np.median(cosines)),
                                           max=max(cosines),negative_count=sum(x<0 for x in cosines),samples=len(cosines)),
        cosine_edge_bands=dict(min=min(band_cosines),median=float(np.median(band_cosines)),
                              max=max(band_cosines),negative_count=sum(x<0 for x in band_cosines)),
        previous_medsam3_msd_archive_edge_weights=med_weights,
        training_scalar_contributions=scalar,
        confirmed_differences=[
            'Conventional gradient-ratio calibration uses detached segmentation logits; SAM3/MedSAM3 uses trainable LoRA parameters.',
            'Conventional Dice-only objective differs from the full SAM3 native loss, which includes classification/presence, boxes and mask terms.',
            'SAM3 semantic foreground multiplies mask probability by query and presence confidences and takes a per-pixel maximum over queries.',
            'GeoFWI3D SAM3 initializes fresh zero-output LoRA; MedSAM3 MSD starts with the medical adapter and uses exclusive three-class probabilities.',
            'GeoFWI3D SAM3 calibrates before any optimizer update and freezes its coefficient; the conventional capped recipe calibrates after a baseline prefix.',
            'Earlier GeoFWI3D SwinUNETR runs used literal 0.1 auxiliary coefficients and a 500-update ramp; this SAM3 recipe uses gradient-ratio targets and a 20-update ramp.'
        ],
        hypothesis='Cold-start calibration of confidence-gated edge gradients yields a large, static coefficient with heterogeneous/conflicting slab gradients; its influence may change rapidly during adaptation.',
        limitations=[
            'Scalar loss ratios are not gradient ratios; per-term gradients were not logged during training.',
            'Large coefficients across different models and datasets are not directly comparable as effective strengths.',
            'The audit confirms formula and coefficient-rule equivalence, not the causal explanation of the Dice collapse.',
            'Final adapters were already retired after completed evaluations; no trained-model confidence-path probe was performed.'
        ],
        next_diagnostic='A short instrumented training-prefix replay on training support only, measuring per-term gradient ratios and cosines and mask/query/presence contributions as learning progresses, with no baseline evaluation rerun.')
    path=REPORT/'edge_implementation_audit.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['training_scalar_contributions','same_loss_source_sha256']},indent=2))


if __name__ == '__main__':
    main()
