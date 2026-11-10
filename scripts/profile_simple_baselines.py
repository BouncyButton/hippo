"""Measure a short predeclared list of classical CNNs at the current FLOP cap."""
import argparse
import hashlib
import json
from pathlib import Path

from profile_intermediate_baselines import audit, intermediate_supplement
from baselines.simple_segmentation import CANDIDATES, build_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(CANDIDATES), required=True)
    parser.add_argument('--budget-flops', type=int, default=94603780000000)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established profiling runtime')
    audit.torch.set_num_threads(4)
    audit.reference.supplementary_flops = intermediate_supplement
    audit.matched.build_network = lambda name, recipe: (build_candidate(args.model), {})
    profiles = []
    for geometry in ['ellipsoid', 'box']:
        row = audit.profile_arm('dice', geometry)
        profiles.append(row)
        print(json.dumps({'model': args.model, 'geometry': geometry, 'cost': row['flops_per_update'], 'parameters': row['parameters']}), flush=True)
    if len({r['flops_per_update'] for r in profiles}) != 1:
        raise ValueError('Target geometry changed the declared arithmetic count')
    budget = args.budget_flops
    if budget < 1:
        raise ValueError('Budget must be positive')
    schedule = audit.schedule(profiles[0]['flops_per_update'], budget)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'baselines/simple_segmentation.py',
               root / 'scripts/profile_intermediate_baselines.py', Path(audit.__file__),
               audit.PROFILE_ROOT / 'profile_reference.py', audit.PROFILE_ROOT / 'profile_matched_models.py',
               root / 'scripts/compute_constraint_objectives.py']
    result = {'model': args.model, 'candidate': CANDIDATES[args.model], 'parameters': profiles[0]['parameters'],
              'schedule': schedule, 'budget_flops': budget, 'profiles': profiles,
              'runtime': {'torch': audit.torch.__version__, 'monai': audit.monai.__version__},
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'status': 'profiled_only', 'dice_target': '70–80% is an empirical development criterion, not inferred from architecture or FLOPs.'}
    output = args.output_dir or root / 'reports/simple_baseline_screen_20261005'
    output.mkdir(parents=True, exist_ok=True)
    (output / (args.model + '_profiles.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'parameters': result['parameters'], 'schedule': schedule}), flush=True)


if __name__ == '__main__':
    main()
