"""Freeze the paired 20-run study using the previously verified MSD splits."""
import hashlib
import json
from pathlib import Path
import random
from scripts.depth_supervision.utils import atomic_json, digest
from .models import ARMS

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/experiments/unwrapped_swin_20261006'


def main():
    p = json.loads((ROOT / 'docs/experiments/depth_swinunetr_20261006/protocol.json').read_text())
    for key in ('sdf_objective', 'layer_sdf', 'calibration', 'source_sha256'):
        p.pop(key)
    p.update(name='unwrapped_swin_fold0_seed0_20261006_01', arms=list(ARMS), arm_settings=ARMS,
             pkl='/home/3160552/hippo/datasets/Dataset101_MSD/msd_hippocampus_full.pkl',
             main_runs=20, preparation_runs=4, guide_epochs=25,
             permitted_auxiliaries=['original depth-balanced multiclass CE'],
             guide_splits={}, fixed_depth_coefficients={}, coefficient_provenance={},
             atlas=dict(margin_mm=8., chart_step=.75, depth_step_mm=1., max_cells_per_chart=2000000,
                        construction='Original cut-shell harmonic disk charts, nearest-surface UV and absolute cutting-plane depth; sides and extra patches separate.',
                        inputs=['normalized MRI', 'longitudinal coordinate / 32 mm (position arms only)',
                                'signed cutting-plane depth / 16 mm (position arms only)', 'sampling support'],
                        annotation_inputs=False, invertible_volume_warp=False,
                        control='Identical MRI samples, chart membership, native return indices, tensor sizes, encoder and fusion; plane UV replaces harmonic UV.',
                        caveat='Surface projection and rasterization can merge samples. Record collisions, intensity error, distortion and tissue coverage.'),
             guide_policy='Two out-of-fold Dice SwinUNETRs per data regime predict training cases; fixed epoch-25 main Dice model predicts validation cases. No held-out labels select guide weights.',
             branch_loss='Dice + frozen original depth-balanced CE; Dice-only epochs 1-5, ramp epochs 6-10. No SDF, focal weighting, topology or auxiliary branch losses.',
             resources=dict(partition='stud', gpu='4g.40gb:1', cpus=4, memory='24G', time='23:50:00'),
             required_dependencies={'monai': '1.6.0', 'libigl': '2.6.1', 'trimesh': '4.7.4', 'scikit-image': '0.26.0'},
             comparisons=['depth versus dice', 'each unfolded arm versus its matched plane arm',
                          'context versus local', 'position versus context', 'attention versus position'],
             evaluation='Native-space mean per-patient A/P Dice, foreground Dice, FP/FN/A-P swaps, HD95, mean surface distance; paired changes and corrections/regressions within each signed depth bin. Single fold/seed exploratory study.')
    for fraction, source in [('5pct', 'low_data'), ('full', 'full_data')]:
        names = p['splits'][fraction]['train'].copy()
        random.Random(p['seed']).shuffle(names)
        half = len(names) // 2
        groups = [names[:half], names[half:]]
        p['guide_splits'][fraction] = [dict(train=sorted(groups[1-k]), predict=sorted(groups[k])) for k in (0, 1)]
        path = ROOT / f'evaluation_output/depth_swinunetr_20261006/{source}_completed_snapshot.json'
        coefficient = json.loads(path.read_text())['runs']['depth']['calibration']['coefficients']['depth']
        p['fixed_depth_coefficients'][fraction] = coefficient
        p['coefficient_provenance'][fraction] = dict(file=str(path.relative_to(ROOT)), sha256=digest(path),
                                                    source='Original depth-only training-case logit-gradient calibration')
    sources = sorted((ROOT / 'scripts/unwrapped_swin').glob('*.py')) + [ROOT / 'scripts/unwrapped_swin/run.sh']
    sources += [ROOT / f'scripts/depth_supervision/{name}.py' for name in ('__init__', 'train', 'utils', 'losses', 'geometry')]
    sources += [ROOT / 'baselines/swin_unetr/swin_unetr.py']
    p['source_sha256'] = {str(f.relative_to(ROOT)): digest(f) for f in sources}
    p['identity_sha'] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True, exist_ok=True)
    atomic_json(DOC / 'protocol.json', p)
    print(json.dumps(dict(protocol=str(DOC / 'protocol.json'), identity=p['identity_sha'],
                         main_runs=p['main_runs'], preparation_runs=p['preparation_runs'])))


if __name__ == '__main__':
    main()
