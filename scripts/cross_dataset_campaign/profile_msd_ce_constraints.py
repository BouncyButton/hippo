"""Remeasure all four MSD Dice+CE models and all fourteen constraint graphs."""
import argparse
import fcntl
from functools import partial
from pathlib import Path

from scripts.cross_dataset_campaign import profile_msd_nnunet as profiler
from scripts.cross_dataset_campaign import msd_ce_constraints as objective
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if args.output.exists():
        existing = read(args.output)
        if (existing.get('status') == 'profiled' and existing.get('model') == args.model
                and existing.get('objective_sha256') == digest(Path(objective.__file__))):
            print('Verified existing complete profile:', args.model, flush=True)
            return
        raise ValueError('Existing profile does not match the current objective')
    audit = profiler.audit
    if audit.monai.__version__ != '1.6.0' or not audit.torch.__version__.startswith('2.2.2'):
        raise ValueError('Use the established profiling runtime')
    audit.torch.set_num_threads(2)
    audit.reference.supplementary_flops = profiler.intermediate_supplement
    profiler.dice_only_optimizer = objective.experiment.ORIGINAL_LOSS_OPTIMIZER
    profiler.calibration_case = objective.calibration_case
    origin = profiler.ROOT/'experiments/cross_dataset_full_20261014/protocols'/f'MSD_{args.model}_f0_target10'
    architecture, protocol = read(origin/'nnunet_architecture.json'), read(origin/'protocol.json')
    rows, costs = [], {}
    for arm in (*ARMS, 'dice_edge', 'dice_edge_sdf'):
        profiler.training_loss = partial(objective.training_loss,
            sdf_policy=protocol['sdf_policy'].get(arm, 'delayed_5_of_9'))
        for calibration in ([False] if arm == 'dice' else [False, True]):
            measured = []
            for geometry in ('ellipsoid', 'box'):
                row = profiler.profile(args.model, arm, geometry, calibration, architecture, protocol)
                rows.append(row); measured.append(row['flops'])
                print({k:row[k] for k in ('model','arm','geometry','calibration','flops')}, flush=True)
            if len(set(measured)) != 1:
                raise ValueError('Geometry changed arithmetic')
            cost = costs.setdefault(arm, dict(calibration_flops=0))
            cost['calibration_flops' if calibration else 'update_flops'] = (
                10*measured[0]+(4096 if 'depth' in arm else 1024) if calibration else measured[0])
        write(args.output.with_suffix('.partial.json'), dict(costs=costs, profiles=rows))
    reference = profiler.ROOT/'reports/msd_dice_ce_bands_pilot_20261014/profiles'/f'{args.model}.json'
    for arm in ('dice','dice_bands'):
        if costs[arm] != read(reference)['costs'][arm]:
            raise ValueError('Established baseline/bands FLOPs changed')
    write(args.output, dict(status='profiled', model=args.model, costs=costs, profiles=rows,
        parameters=rows[0]['parameters'], input_shape=[1,1,64,64,64], calibration_base='Dice+CE',
        baseline_bands_reference_sha256=digest(reference), objective_sha256=digest(Path(objective.__file__)),
        accounting='Established dense MAC=2; optimization cap; calibration measured separately; original SDF timing preserved',
        runtime=dict(torch=audit.torch.__version__, monai=audit.monai.__version__)))


if __name__ == '__main__':
    main()
