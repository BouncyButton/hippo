"""Profile background-including/excluding Dice with the reference audit runtime."""
from functools import partial
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import profile_unet_constraint_budget as audit
import compute_constraint_objectives as objectives
from monai.losses import DiceLoss
from train_unet48_budget_pilot import KWARGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resize-decoder', action='store_true')
    group.add_argument('--linear-head', action='store_true')
    args = parser.parse_args()
    assert audit.monai.__version__ == '1.6.0'
    assert audit.torch.__version__.startswith('2.2.2')
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = audit.constraint_supplement
    if args.linear_head:
        from train_linear_head import build_linear_head_model
        audit.matched.build_network = lambda name, recipe: (build_linear_head_model(), {})
    elif args.resize_decoder:
        from train_resize_decoder import build_resize_model
        audit.matched.build_network = lambda name, recipe: (build_resize_model(), {})
    else:
        audit.matched.build_network = lambda name, recipe: (audit.matched.UNet(**KWARGS), {})
    profiles = []
    for background in [True, False]:
        objectives.DiceLoss = partial(DiceLoss, include_background=background)
        for geometry in ['ellipsoid', 'box']:
            row = audit.profile_arm('dice', geometry)
            row['include_background'] = background
            profiles.append(row)
            print(json.dumps({k: row[k] for k in ['geometry', 'include_background', 'flops_per_update']}), flush=True)
        assert len({r['flops_per_update'] for r in profiles if r['include_background'] == background}) == 1
    costs = {str(background): next(r['flops_per_update'] for r in profiles if r['include_background'] == background)
             for background in [True, False]}
    if not args.resize_decoder and not args.linear_head:
        assert costs['True'] == 78849850205
    files = [Path(__file__), ROOT/'scripts/unet48_stability/train_stability.py',
             ROOT/'scripts/compute_constraint_objectives.py', ROOT/'scripts/profile_unet_constraint_budget.py',
             audit.PROFILE_ROOT/'profile_reference.py', audit.PROFILE_ROOT/'profile_matched_models.py']
    if args.resize_decoder:
        files.append(Path(__file__).with_name('train_resize_decoder.py'))
    if args.linear_head:
        files.append(Path(__file__).with_name('train_linear_head.py'))
    report = {'runtime': {'torch': audit.torch.__version__, 'monai': audit.monai.__version__},
              'costs': costs, 'profiles': profiles, 'resize_decoder': args.resize_decoder,
              'linear_head': args.linear_head,
              'accounting': 'Same declared arithmetic audit as official B_ref; all training operations, excluding validation/augmentation/I/O.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    output = ROOT/'reports/unet48_stability_20261005'
    output.mkdir(parents=True, exist_ok=True)
    filename = 'linear_profiles.json' if args.linear_head else 'resize_profiles.json' if args.resize_decoder else 'profiles.json'
    (output/filename).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
