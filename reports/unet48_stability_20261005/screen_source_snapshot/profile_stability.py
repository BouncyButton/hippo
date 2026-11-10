"""Profile background-including/excluding Dice with the reference audit runtime."""
from functools import partial
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
    assert audit.monai.__version__ == '1.6.0'
    assert audit.torch.__version__.startswith('2.2.2')
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = audit.constraint_supplement
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
    assert costs['True'] == 78849850205
    files = [Path(__file__), ROOT/'scripts/unet48_stability/train_stability.py',
             ROOT/'scripts/compute_constraint_objectives.py', ROOT/'scripts/profile_unet_constraint_budget.py',
             audit.PROFILE_ROOT/'profile_reference.py', audit.PROFILE_ROOT/'profile_matched_models.py']
    report = {'runtime': {'torch': audit.torch.__version__, 'monai': audit.monai.__version__},
              'costs': costs, 'profiles': profiles,
              'accounting': 'Same declared arithmetic audit as official B_ref; all training operations, excluding validation/augmentation/I/O.',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    output = ROOT/'reports/unet48_stability_20261005'
    output.mkdir(parents=True, exist_ok=True)
    (output/'profiles.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
